import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from utils.generate_utils import mask_for_de_novo, calculate_cosine_sim, calculate_hamming_dist
from diffusion import Diffusion
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from helm_tokenizer.helm_tokenizer import HelmTokenizer
from utils.helm_utils import create_helm_from_aa_seq, get_smi_from_helms
from utils.filter import PeptideAnalyzer
from new_tokenizer.ape_tokenizer import APETokenizer
from scoring.scoring_functions import ScoringFunctions


@torch.no_grad()
def generate_sequence_unconditional(config, sequence_length: int, mdlm: Diffusion):
    tokenizer = mdlm.tokenizer
    # generate array of [MASK] tokens
    masked_array = mask_for_de_novo(config, sequence_length)

    if config.vocab == 'old_smiles':
        # use custom encode function
        inputs = tokenizer.encode(masked_array)
    elif config.vocab == 'new_smiles' or config.vocab == 'selfies':
        inputs = tokenizer.encode_for_generation(masked_array)
    else:
        # custom HELM tokenizer
        inputs = tokenizer(masked_array, return_tensors="pt")
    
    # tokenized masked array
    inputs = {key: value.to(mdlm.device) for key, value in inputs.items()}
    # sample unconditional array of tokens
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness

    return logits, inputs


@hydra.main(version_base=None, config_path='/home/st512/peptune/scripts/peptide-mdlm-mcts', config_name='config')
def main(config):
    path = "/home/st512/peptune/scripts/peptide-mdlm-mcts"

    if config.vocab == 'new_smiles':
        tokenizer = APETokenizer()
        tokenizer.load_vocabulary('/home/st512/peptune/scripts/peptide-mdlm-mcts/new_tokenizer/peptide_smiles_600_vocab.json')
    elif config.vocab == 'old_smiles':
        tokenizer = SMILES_SPE_Tokenizer('/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_vocab.txt', 
                                    '/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_splits.txt')
    elif config.vocab == 'selfies':
        tokenizer = APETokenizer()
        tokenizer.load_vocabulary('/home/st512/peptune/scripts/peptide-mdlm-mcts/new_tokenizer/peptide_selfies_600_vocab.json')
    elif config.vocab == 'helm':
        tokenizer = HelmTokenizer('/home/st512/peptune/scripts/peptide-mdlm-mcts/helm_tokenizer/monomer_vocab.txt')
    
    mdlm_model = Diffusion.load_from_checkpoint(config.eval.checkpoint_path, config=config, tokenizer=tokenizer, strict=False)
    
    mdlm_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mdlm_model.to(device)

    print("loaded models...")
    analyzer = PeptideAnalyzer()
    
    gfap = 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM'

    # scoring functions
    score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'permeability']
    score_functions = ScoringFunctions(score_func_names, [gfap])
    

    max_seq_length = config.sampling.seq_length
    num_sequences = config.sampling.num_sequences
    generation_results = []
    num_valid = 0.
    num_total = 0.
    while num_total < num_sequences: 
        num_total += 1
        generated_array, input_array = generate_sequence_unconditional(config, max_seq_length, mdlm_model)
        
        # store in device
        generated_array = generated_array.to(mdlm_model.device)
        print(generated_array)
        
        # compute masked perplexity
        perplexity = mdlm_model.compute_masked_perplexity(generated_array, input_array['input_ids'])
        perplexity = round(perplexity, 4)
        
        if config.vocab == 'old_smiles' or config.vocab == 'new_smiles':
            smiles_seq = tokenizer.decode(generated_array)
            if analyzer.is_peptide(smiles_seq):
                aa_seq, seq_length = analyzer.analyze_structure(smiles_seq)
                num_valid += 1
                scores = score_functions(input_seqs=[smiles_seq])
                
                binding = scores[0][0]
                sol = scores[0][1]
                hemo = scores[0][2]
                nf = scores[0][3]
                perm = scores[0][4]
                
                generation_results.append([smiles_seq, perplexity, aa_seq, binding, sol, hemo, nf, perm])
            else:
                aa_seq = "not valid peptide"
                seq_length = '-'
                scores = "not valid peptide"
        elif config.vocab == 'selfies':
            smiles_seq = tokenizer.decode(generated_array)
        else:
            aa_seq = tokenizer.decode(generated_array)
            smiles_seq = get_smi_from_helms(aa_seq)
        
        
        print(f"perplexity: {perplexity} | length: {seq_length} | smiles sequence: {smiles_seq} | amino acid sequence: {aa_seq} | scores: {scores}")
        sys.stdout.flush()

    valid_frac = num_valid / num_total
    print(f"fraction of synthesizable peptides: {valid_frac}")
    df = pd.DataFrame(generation_results, columns=['Generated SMILES', 'Perplexity', 'Peptide Sequence', 'Binding Affinity', 'Solubility', 'Hemolysis', 'Nonfouling', 'Permeability'])
    df.to_csv(path + f'/benchmarks/unconditional/epoch-10-pretrain-gfap.csv', index=False)
        
if __name__ == "__main__":
    main()