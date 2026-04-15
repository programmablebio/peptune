import os
import torch
import torch.nn.functional as F
import sys
import pandas as pd
import omegaconf
from utils.generate_utils import mask_for_de_novo, calculate_cosine_sim, calculate_hamming_dist
from diffusion import Diffusion
import hydra
from tqdm import tqdm
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from utils.app import PeptideAnalyzer
from scoring.scoring_functions import ScoringFunctions

# Register custom OmegaConf resolvers required by config.yaml
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
omegaconf.OmegaConf.register_new_resolver('eval', eval, replace=True)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)

base_path = '/path/to/your/home/PepTune'
ckpt_path = base_path + '/checkpoints/peptune-pretrained.ckpt'

@torch.no_grad()
def generate_sequence_unconditional(config, sequence_length: int, mdlm: Diffusion):
    tokenizer = mdlm.tokenizer
    # generate array of [MASK] tokens
    masked_array = mask_for_de_novo(config, sequence_length)

    inputs = tokenizer.encode(masked_array)
    
    # tokenized masked array
    inputs = {key: value.to(mdlm.device) for key, value in inputs.items()}
    # sample unconditional array of tokens
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness

    return logits, inputs


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config):

    tokenizer = SMILES_SPE_Tokenizer(f'{base_path}/src/tokenizer/new_vocab.txt', 
                                f'{base_path}/src/tokenizer/new_splits.txt')

    # Build model with current config, then load weights manually
    # (load_from_checkpoint overrides config with saved hparams)
    mdlm_model = Diffusion(config=config, tokenizer=tokenizer)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mdlm_model.load_state_dict(ckpt["state_dict"], strict=False)
    
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
        
        
        print(f"perplexity: {perplexity} | length: {seq_length} | smiles sequence: {smiles_seq} | amino acid sequence: {aa_seq} | scores: {scores}")
        sys.stdout.flush()

    valid_frac = num_valid / num_total
    print(f"fraction of synthesizable peptides: {valid_frac}")
    df = pd.DataFrame(generation_results, columns=['Generated SMILES', 'Perplexity', 'Peptide Sequence', 'Binding Affinity', 'Solubility', 'Hemolysis', 'Nonfouling', 'Permeability'])
    df.to_csv(base_path + f'/results/test_generate.csv', index=False)
        
if __name__ == "__main__":
    main()