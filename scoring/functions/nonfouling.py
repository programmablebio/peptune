import sys
import os
sys.path.append('/home/st512/peptune/scripts/peptide-mdlm-mcts')
import xgboost as xgb
import torch
import numpy as np
from transformers import AutoModelForMaskedLM
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
import warnings
import numpy as np
from rdkit import Chem, rdBase, DataStructs
from transformers import AutoModelForMaskedLM


rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Nonfouling:
    
    def __init__(self):
        self.predictor = xgb.Booster(model_file='/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/nonfouling/new_data/best_model.json')
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer
        self.tokenizer = SMILES_SPE_Tokenizer('/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_vocab.txt', 
                                              '/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_splits.txt')
    
    def generate_embeddings(self, sequences):
        embeddings = []
        for sequence in sequences:
            tokenized = self.tokenizer(sequence, return_tensors='pt')
            with torch.no_grad():
                output = self.emb_model(**tokenized)
            # Mean pooling across sequence length
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def get_scores(self, input_seqs: list):
        scores = np.zeros(len(input_seqs))
        features = self.generate_embeddings(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        # return the probability of it being not hemolytic
        return scores
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores
    
def unittest():
    nf = Nonfouling()
    seq = ["NCC(=O)N[C@H](CS)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](CC1=CN=C-N1)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CC(=CN2)C1=C2C=CC=C1)C(=O)N[C@@H](c1ccc(cc1)F)C(=O)N[C@@H]([C@H](CC)C)C(=O)N[C@@H](CCCO)C(=O)N[C@@H](CC1=CN=C-N1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CO)C(=O)O"]

    scores = nf(input_seqs=seq)
    print(scores)


if __name__ == '__main__':
    unittest()