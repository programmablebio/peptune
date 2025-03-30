import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import random as rd

from diffusion import Diffusion
from scoring.scoring_functions import ScoringFunctions
from utils.filter import PeptideAnalyzer
import noise_schedule

""""
    Notes: store rolled out sequence?
    path of node objects or strings?
    should we only select valid expandable leaf nodes?
    calculate similarity between sibling nodes?
    should we evaluate generated sequences?
"""
class Node:
    """
        Node class: partially unmasked SMILES string
        - parentNode: Node object at previous time step
        - childNodes: set of M Node objects generated from sampling M distinct unmasking schemes
        - totalReward: vector of cumulative rewards for all K objectives
        - visits: number of times the node has been visited by an interation
        - path: array of partially unmasked SMILES strings leading to the node from the completely masked root node
        - timestep: the time step where the sequence was sampled
        - sampleProb: probability of sampling the sequence from the diffusion model
    """
    def __init__(self, config, tokens=None, parentNode=None, childNodes=[], scoreVector=None, totalReward=None, timestep=None, sampleProb=None):
        self.config = config 
        self.parentNode = parentNode
        self.childNodes = childNodes
        self.scoreVector = scoreVector
        
        # initialize total rewards to the reward of the roll out unmasked sequence
        if totalReward is not None:
            self.totalReward = totalReward
        else:
            self.totalReward = np.zeros(self.config.mcts.num_objectives)
            
        # set initial visits to 1
        self.visits = 1
        # array of all sequences in path from the root -> node
        #self.path = path 
        # set timestep (value between 0 and num_steps)
        self.timestep = timestep
        # set the sampling probabiltiy equal to the probability from the reverse posterior
        self.sampleProb = sampleProb
        
        # dict with 'input_ids' as token array and 'attention_mask' 
        self.tokens = tokens
        
        #self.sequence = sequence
    
    def selectNode(self, num_func):
        """
            Selects a node to move to among the children nodes
        """
        # extract the status of the current node
        nodeStatus = self.getExpandStatus()
        
        # if the node is a legal non-leaf node
        if (nodeStatus == 3):
            # initialize array that will store select score vectors of each child node
            paretoFront = {}
            for childNode in self.childNodes:
                childStatus = childNode.getExpandStatus()
                # only append child if it is legal leaf node (expandable) or legal non-leaf node
                if childStatus == 2 or childStatus == 3:
                    selectScore = childNode.calcSelectScore()
                    paretoFront = updateParetoFront(paretoFront, childNode, selectScore, num_func)
            
            # randomly select a node on the Pareto front
            #selected = rd.choice(paretoFront)
            selected = rd.choice(list(paretoFront.keys()))
            # return selected child node and status
            return selected, selected.getExpandStatus()
        
        # if node is not valid non-leaf node
        return self, nodeStatus

    def addChildNode(self, tokens, totalReward, prob=None):
        """"
            Adds a child node
        """
        child = Node(config=self.config,
                     tokens=tokens, 
                     parentNode=self, 
                     childNodes=[], 
                     totalReward=totalReward,
                     timestep=self.timestep+1,
                     sampleProb=prob)
        
        self.childNodes.append(child)
        return child
    
    def updateNode(self, rewards):
        """
            Updates the cumulative rewards vector with the reward vector at a descendent leaf node. 
            Increments the number of visits to the node.
        """
        self.visits += 1
        self.totalReward += rewards
    
    def calcSelectScore(self):
        """
            Calculates the select score for the node from the cumulative rewards vector and number of visits.
            - c: determines the degree of exploration
            - minSelectScore: determines the 
        """
        """"
        if not self.parentNode:
            return 0.0
        """
        # K-dimensional vector of normalized rewards for each objective 
        normRewards = self.totalReward / self.visits 
        if self.sampleProb is not None:
            print("Sample Prob")
            print(self.sampleProb)
            return normRewards + (self.config.mcts.sample_prob * self.sampleProb * np.sqrt(self.root.visits) / self.visits)
        return normRewards
    
    def getExpandStatus(self):
        """
            Returns an integer indicating whether the node is a:
            1. terminal node (sequence is fully unmasked)
            2. legal leaf node (partially unmasked sequence that can be expanded)
            3. legal non-leaf node (already expanded sequence with M child nodes)
        """
        if self.timestep == self.config.sampling.steps:
            return 1
        elif (self.timestep < self.config.sampling.steps) and (len(self.childNodes) == 0):
            return 2
        return 3
    
"""END OF NODE CLASS"""

def updateParetoFront(paretoFront, node, scoreVector, num_func):
    """
        Removes sequences that are dominated by scoreVector
        adds the SMILES sequence if it is non-dominated and its scoreVector
    """
    paretoSize = len(paretoFront)
    if paretoSize == 0:
        # if pareto front is empty, add sequence and scoreVector
        paretoFront[node] = scoreVector
    else:
        # vector of boolean
        # true: sequence is non-dominated by the pareto-optimal sequence
        # false: sequence is completely dominated by the pareto-optimal sequence
        nondominate = []
        # sequences to be deleted
        delete = []
        for k, v in paretoFront.items():
            nondominated = scoreVector >= np.asarray(v)
            dominant = scoreVector > np.asarray(v)
            
            if num_func <= len(nondominated):
                attn_nondominated = nondominated[:num_func]
                attn_dominant = dominant[:num_func]
            
            # all scores are greater than or equal to v and at least one score is strictly greater than v
            if attn_nondominated.all() and attn_dominant.any():
                # add the dominated sequence to be deleted
                delete.append(k)
                # sequence is dominant
                nondominate.append(True)
            elif attn_nondominated.all():
                # sequence is non-dominated
                nondominate.append(True)
            else:
                # sequence is completely dominated
                nondominate.append(False)
        
        nondominate = np.asarray(nondominate)
        # if sequence is either dominant or non-dominated by all sequences in pareto-front -> add to pareto front
        if nondominate.all():
            paretoFront[node] = scoreVector
            
        # delete all dominated sequences
        while (paretoSize > 0) and (len(delete) > 0):
            #for k in delete:
            del paretoFront[delete[0]]
            del delete[0]
            paretoSize -= 1
    return paretoFront
    
"""BEGINNING OF MCTS CLASS"""

class MCTS:
    def __init__(self, config, max_sequence_length=None, mdlm=None, score_func_names=[], prot_seqs=None, num_func = []):
        self.config = config
        self.noise = noise_schedule.get_noise(config)
        self.time_conditioning = self.config.time_conditioning
        # dictionary of k (SMILES string) and v (score vector) of Pareto-optimal sequences
        self.peptideParetoFront = {} 
        self.num_steps = config.sampling.steps
        self.num_sequences = config.sampling.num_sequences
        
        # mdlm model
        self.mdlm = mdlm
        self.tokenizer = mdlm.tokenizer
        self.device = mdlm.device
        
        if max_sequence_length is None:
            self.sequence_length = self.config.sampling.seq_length
        else:
            self.sequence_length = max_sequence_length
            
        self.num_iter = config.mcts.num_iter
        
        self.num_child = config.mcts.num_children
        
        # score functions
        self.score_functions = ScoringFunctions(score_func_names, prot_seqs)
        self.num_func = num_func # K-dimensional vector with the iteration number to start conditioning on each of the objectives in increasng order
        self.iter_num = 0
        self.curr_num_func = 1
        self.analyzer = PeptideAnalyzer()
        
        # track fraction of valid peptides
        self.valid_fraction_log = []
        self.affinity1_log = []
        self.affinity2_log = []
        self.permeability_log = []
        self.sol_log = []
        self.hemo_log = []
        self.nf_log = []
        
    def reset(self):
        self.iter_num = 0
        self.valid_fraction_log = []
        self.affinity1_log = []
        self.affinity2_log = []
        self.permeability_log = []
        self.sol_log = []
        self.hemo_log = []
        self.nf_log = []
        self.peptideParetoFront = {} 
        
    def forward(self, rootNode):
        self.reset()
        
        while (self.iter_num < self.num_iter):
            self.iter_num += 1
            
            # traverse the tree form the root node until a leaf node
            leafNode, _ = self.select(rootNode)
            #print(leafNode.tokens['input_ids'])
            
            # expand leaf node into num_children partially unmasked sequences at the next timestep
            self.expand(leafNode)
        
        # return dictionary of pareto front peptides and their score vectors
        return self.peptideParetoFront

    # change to include more even if dominated? since there is error in the scores
    def updateParetoFront(self, sequence, scoreVector, tokens):
        """
            Removes sequences that are dominated by scoreVector
            adds the SMILES sequence if it is non-dominated and its scoreVector
            
            num_func: index of the last objective to consider when updating the pareto front from 0 to K
        """
        paretoSize = len(self.peptideParetoFront)
        
        self.curr_num_func = 1
                
        for i in range(len(self.num_func)):
            if self.iter_num >= self.num_func[i]:
                self.curr_num_func = i+1
        
        if paretoSize == 0:
            # if pareto front is empty, add sequence and scoreVector
            self.peptideParetoFront[sequence] = {'scores': scoreVector, 'token_ids': tokens}
            # if pareto front is empty, set reward vector to 1s
            rewardVector = np.ones(len(scoreVector))
        else:
            # vector of boolean
            # true: sequence is non-dominated by the pareto-optimal sequence
            # false: sequence is completely dominated by the pareto-optimal sequence
            nondominate = []
            # sequences to be deleted
            delete = []
            # initialize reward vector with zeros
            rewardVector = np.zeros(len(scoreVector))
            for k, v in self.peptideParetoFront.items():
                 # boolean vector 
                # true: if all metrics are equal or larger
                # false: if the pareto front sequence dominates scoreVector
                nondominated = scoreVector >= np.asarray(v['scores']) # [num_objectives]
                dominant = scoreVector > np.asarray(v['scores'])
                # add to reward vector
                rewardVector += nondominated # [num_objectives]

                if self.curr_num_func <= len(nondominated):
                    attn_nondominated = nondominated[:self.curr_num_func]
                    attn_dominant = dominant[:self.curr_num_func]
                
                # only delete pareto-optimal sequence if
                # all scores are greater than or equal to v and at least one score is strictly greater than v
                if attn_nondominated.all() and attn_dominant.any():
                    # add the dominated sequence to be deleted
                    delete.append(k)
                    # sequence is dominant
                    nondominate.append(True)
                elif attn_nondominated.all():
                    # sequence is non-dominated
                    nondominate.append(True)
                else:
                    # sequence is completely dominated
                    nondominate.append(False)
            
            assert len(nondominate) == paretoSize
            nondominate = np.asarray(nondominate)
            # if sequence is either dominant or non-dominated by all sequences in pareto-front -> add to pareto front
            # or if the pareto front does not have enough sequences
            if nondominate.all() or paretoSize < self.num_sequences:
                self.peptideParetoFront[sequence] = {'scores': scoreVector, 'token_ids': tokens}
            
            rewardVector = rewardVector / paretoSize
                
            # delete all dominated sequences if pareto front is larger than num_sequences
            while (paretoSize > self.num_sequences) and (len(delete) > 0):
                #for k in delete:
                del self.peptideParetoFront[delete[0]]
                del delete[0]
                paretoSize -= 1
            
        return rewardVector

    def isPathEnd(self, path, maxDepth): 
        """
            Checks if the node is completely unmasked (ie. end of path)
            or if the path is at the max depth
        """
        if (path[-1] != self.config.mcts.mask_token).all():
            return True
        elif len(path) >= maxDepth: 
            return True
        return False
    
    def select(self, currNode):
        """
            Traverse the tree from the root node until reaching a legal leaf node
        """
        while True: 
            currNode, nodeStatus = currNode.selectNode(self.curr_num_func)
            if nodeStatus != 3:
                return currNode, nodeStatus
            
    def expand(self, parentNode, eps=1e-5, checkSimilarity = True):
        """
            Sample unmasking steps from the pre-trained MDLM 
            adds num_children partially unmasked sequences to the children of the parentNode
        """
        
        num_children = self.config.mcts.num_children
        # initialize child rewards that will be added to total rewards
        allChildReward = np.zeros_like(parentNode.totalReward) # (n_objectives)
        
        
        # compute number of rollout steps
        # if parentNode.timestep = self.num_steps then num_rollout_steps = 1
        num_rollout_steps = self.num_steps - parentNode.timestep
        # array of rollout timesteps from the timestep of parent node to 0
        rollout_t = torch.linspace(1, eps, num_rollout_steps, device=self.device)
        dt = (1 - eps) / self.num_steps
        p_x0_cache = None
        
        # initialize x and attn_mask
        x = parentNode.tokens['input_ids'].to(self.device)
        attn_mask = parentNode.tokens['attention_mask'].to(self.device)
        
        t = rollout_t[0] * torch.ones(num_children, 1, device = self.device)
        # generate (n_children, seq_length) array of sampled children nodes
        print("token array:")
        print(x)
        p_x0_cache, x_children = self.mdlm.batch_cached_reverse_step(token_array=x, 
                                                         t=t, dt=dt, 
                                                         batch_size=num_children, 
                                                         attn_mask=attn_mask)
        x_rollout = x_children
        
        for i in range(1, num_rollout_steps):
            t = rollout_t[i] * torch.ones(num_children, 1, device = self.device)
            
            p_x0_cache, x_next = self.mdlm.cached_reverse_step(x=x_rollout, 
                                                               t=t, dt=dt, p_x0=p_x0_cache, 
                                                               attn_mask=attn_mask)
            
            if (not torch.allclose(x_next, x) or self.time_conditioning):
                # Disable caching
                p_x0_cache = None
                
            x_rollout = x_next
                
        if self.config.sampling.noise_removal:
            t = rollout_t[-1] * torch.ones(x.shape[0], 1, device=self.device)
            """if self.sampler == 'analytic':
                x = self.mdlm._denoiser_update(x, t)
            else:"""
            time_cond = self.noise(t)[0]
            x_rollout = self.mdlm.forward(x_rollout, attn_mask, time_cond).argmax(dim=-1) # (n_children, seq_length)
        
        childSequences = self.tokenizer.batch_decode(x_rollout)
        
        validSequences = []
        maskedTokens = []
        unmaskedTokens = []
        for i in range(num_children):
            childSeq = childSequences[i]
            #scoreVector = scoreVectors[i]
            rewardVector = np.zeros(self.config.mcts.num_objectives)
            
            # check if the peptide is valid
            if self.analyzer.is_peptide(childSeq):
                validSequences.append(childSeq)
                maskedTokens.append(x_children[i])
                unmaskedTokens.append(x_rollout[i])
            else:
                childTokens = {'input_ids': x_children[i], 'attention_mask': attn_mask}
                parentNode.addChildNode(tokens=childTokens, 
                                totalReward=rewardVector)
        
        if (len(validSequences) != 0):
            scoreVectors = self.score_functions(input_seqs=validSequences) 
            average_scores = scoreVectors.T
            if self.config.mcts.single:
                self.permeability_log.append(average_scores[0])
            else:
                self.affinity1_log.append(average_scores[0])
                self.sol_log.append(average_scores[1])
                self.hemo_log.append(average_scores[2])
                self.nf_log.append(average_scores[3])
                if self.config.mcts.perm: 
                    self.permeability_log.append(average_scores[4])
                elif self.config.mcts.dual: 
                    self.affinity2_log.append(average_scores[4])
        else:
            self.affinity1_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            self.sol_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            self.hemo_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            self.nf_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            
            if self.config.mcts.perm: 
                self.permeability_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            elif self.config.mcts.dual: 
                self.affinity2_log.append(np.zeros((self.config.mcts.num_objectives, self.config.sampling.num_sequences)))
            
        for i, validSeq in enumerate(validSequences):
            #tokens = validTokens[i]
            scoreVector = scoreVectors[i]
            
            # update pareto front
            rewardVector = self.updateParetoFront(validSeq, scoreVector, unmaskedTokens[i])
            print(scoreVector)
            print(rewardVector)
            
            # add to all child reward vector for backprop
            allChildReward += rewardVector
            
            # create node for sequence and add to the children node of parent
            childTokens = {'input_ids': maskedTokens[i], 'attention_mask': attn_mask}
            parentNode.addChildNode(tokens=childTokens, 
                            totalReward=rewardVector)
        
        # compute fraction of invalid child sequences
        invalid = (num_children - len(validSequences)) / num_children

        valid_fraction = len(validSequences) / num_children
        print(f"Valid fraction: {valid_fraction}")
        self.valid_fraction_log.append(valid_fraction)
        
        print(self.config.mcts.invalid_penalty)
        # subtract score using fraction of invalid sequences from reward
        allChildReward = allChildReward - (self.config.mcts.invalid_penalty * invalid)
        # backpropogate all child rewards
        self.backprop(parentNode, allChildReward)


    def backprop(self, node, reward_vector):
        # backpropogate rewards through the path leading to the leaf node from the root
        while node:
            node.updateNode(reward_vector)
            node = node.parentNode
            

    def getSequenceForObjective(self, objective_index, k):
        """
            Returns the top-k sequences in the pareto front that has the best score for
            a given objective and their score vectors for all objectives
        """
        
        # dictionary of top-k peptides for the objective
        topk = {}
        
        peptides = []
        objectiveScores = []
        for k, v in self.peptideParetoFront.items():
            # store peptides in list
            peptides.append(k)
            # store score for objective
            objectiveScores.append(v['token_ids'][objective_index])
        
        objectiveScores = torch.tensor(objectiveScores)
        topKScores = torch.topk(objectiveScores, k)
        for (_, index) in topKScores.items():
            seq = peptides[index]
            
            topk[seq] = self.peptideParetoFront.get(seq)
        
        return topk
            
