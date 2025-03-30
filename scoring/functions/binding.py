import sys
sys.path.append('/home/st512/peptune/scripts/peptide-mdlm-mcts')
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import pandas as pd
import torch.nn as nn
import esm
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig


class ImprovedBindingPredictor(nn.Module):
    def __init__(self, 
                 esm_dim=1280,
                 smiles_dim=768,
                 hidden_dim=512,
                 n_heads=8,
                 n_layers=3,
                 dropout=0.1):
        super().__init__()
        
        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM
        
        # Project to same dimension
        self.smiles_projection = nn.Linear(smiles_dim, hidden_dim)
        self.protein_projection = nn.Linear(esm_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.smiles_norm = nn.LayerNorm(hidden_dim)
        
        # Cross attention blocks with layer norm
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])
        
        # Prediction heads
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Regression head
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (3 classes: tight, medium, loose binding)
        self.classification_head = nn.Linear(hidden_dim, 3)
        
    def get_binding_class(self, affinity):
        """Convert affinity values to class indices
        0: tight binding (>= 7.5)
        1: medium binding (6.0-7.5)
        2: weak binding (< 6.0)
        """
        if isinstance(affinity, torch.Tensor):
            tight_mask = affinity >= self.tight_threshold
            weak_mask = affinity < self.weak_threshold
            medium_mask = ~(tight_mask | weak_mask)
            
            classes = torch.zeros_like(affinity, dtype=torch.long)
            classes[medium_mask] = 1
            classes[weak_mask] = 2
            return classes
        else:
            if affinity >= self.tight_threshold:
                return 0  # tight binding
            elif affinity < self.weak_threshold:
                return 2  # weak binding
            else:
                return 1  # medium binding
        
    def forward(self, protein_emb, smiles_emb):
        protein = self.protein_norm(self.protein_projection(protein_emb))
        smiles = self.smiles_norm(self.smiles_projection(smiles_emb))
        
        #protein = protein.transpose(0, 1)
        #smiles = smiles.transpose(0, 1)
        
        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to SMILES
            attended_protein = layer['attention'](
                protein, smiles, smiles
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))
            
            # SMILES attending to protein
            attended_smiles = layer['attention'](
                smiles, protein, protein
            )[0]
            smiles = layer['norm1'](smiles + attended_smiles)
            smiles = layer['norm2'](smiles + layer['ffn'](smiles))
        
        # Get sequence-level representations
        protein_pool = torch.mean(protein, dim=0)
        smiles_pool = torch.mean(smiles, dim=0)
        
        # Concatenate both representations
        combined = torch.cat([protein_pool, smiles_pool], dim=-1)
        
        # Shared features
        shared_features = self.shared_head(combined)
        
        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)
        
        return regression_output, classification_logits
    
class BindingAffinity:
    def __init__(self, prot_seq, model_type='PeptideCLM'):
        super().__init__()
        
        if model_type == 'PepDoRA':
            # peptide embeddings
            model_name = "ChatterjeeLab/PepDoRA"
            self.pep_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pep_model = AutoModel.from_pretrained(model_name)
            
            self.model = ImprovedBindingPredictor(smiles_dim=384)
            checkpoint = torch.load('/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/binding/best_model_optuna1.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # peptide embeddings
            self.pep_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer
            self.pep_tokenizer = SMILES_SPE_Tokenizer('/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_vocab.txt', 
                                                      '/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_splits.txt')
            
            self.model = ImprovedBindingPredictor()
            checkpoint = torch.load('/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/binding/best_model.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load ESM-2 model
        self.prot_tokenizer = alphabet.get_batch_converter() # load esm tokenizer

        data = [("target", prot_seq)]  
        # get tokenized protein
        _, _, prot_tokens = self.prot_tokenizer(data)
        with torch.no_grad():
            results = self.esm_model.forward(prot_tokens, repr_layers=[33])  # Example with ESM-2
            prot_emb = results["representations"][33]
            
        self.prot_emb = prot_emb[0]
        self.prot_emb = torch.mean(self.prot_emb, dim=0, keepdim=True)
        
    
    def forward(self, input_seqs):        
        with torch.no_grad():
            scores = []
            for seq in input_seqs:
                pep_tokens = self.pep_tokenizer(seq, return_tensors='pt', padding=True)
                
                with torch.no_grad():
                    emb = self.pep_model(input_ids=pep_tokens['input_ids'], 
                                         attention_mask=pep_tokens['attention_mask'], 
                                         output_hidden_states=True)
                    
                #emb = self.pep_model(input_ids=pep_tokens['input_ids'], attention_mask=pep_tokens['attention_mask'])
                pep_emb = emb.last_hidden_state.squeeze(0)
                pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)
                
                score, logits = self.model.forward(self.prot_emb, pep_emb)
                scores.append(score.item())
        return scores
    
    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)

def unittest():
    amhr = 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV'
    tfr = 'MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF'
    gfap = 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM'
    glp1 = 'MAGAPGPLRLALLLLGMVGRAGPRPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVLSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCS'
    glast = 'MTKSNGEEPKMGGRMERFQQGVRKRTLLAKKKVQNITKEDVKSYLFRNAFVLLTVTAVIVGTILGFTLRPYRMSYREVKYFSFPGELLMRMLQMLVLPLIISSLVTGMAALDSKASGKMGMRAVVYYMTTTIIAVVIGIIIVIIIHPGKGTKENMHREGKIVRVTAADAFLDLIRNMFPPNLVEACFKQFKTNYEKRSFKVPIQANETLVGAVINNVSEAMETLTRITEELVPVPGSVNGVNALGLVVFSMCFGFVIGNMKEQGQALREFFDSLNEAIMRLVAVIMWYAPVGILFLIAGKIVEMEDMGVIGGQLAMYTVTVIVGLLIHAVIVLPLLYFLVTRKNPWVFIGGLLQALITALGTSSSSATLPITFKCLEENNGVDKRVTRFVLPVGATINMDGTALYEALAAIFIAQVNNFELNFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWFLDRLRTTTNVLGDSLGAGIVEHLSRHELKNRDVEMGNSVIEENEMKKPYQLIAQDNETEKPIDSETKM'
    ncam = 'LQTKDLIWTLFFLGTAVSLQVDIVPSQGEISVGESKFFLCQVAGDAKDKDISWFSPNGEKLTPNQQRISVVWNDDSSSTLTIYNANIDDAGIYKCVVTGEDGSESEATVNVKIFQKLMFKNAPTPQEFREGEDAVIVCDVVSSLPPTIIWKHKGRDVILKKDVRFIVLSNNYLQIRGIKKTDEGTYRCEGRILARGEINFKDIQVIVNVPPTIQARQNIVNATANLGQSVTLVCDAEGFPEPTMSWTKDGEQIEQEEDDEKYIFSDDSSQLTIKKVDKNDEAEYICIAENKAGEQDATIHLKVFAKPKITYVENQTAMELEEQVTLTCEASGDPIPSITWRTSTRNISSEEKASWTRPEKQETLDGHMVVRSHARVSSLTLKSIQYTDAGEYICTASNTIGQDSQSMYLEVQYAPKLQGPVAVYTWEGNQVNITCEVFAYPSATISWFRDGQLLPSSNYSNIKIYNTPSASYLEVTPDSENDFGNYNCTAVNRIGQESLEFILVQADTPSSPSIDQVEPYSSTAQVQFDEPEATGGVPILKYKAEWRAVGEEVWHSKWYDAKEASMEGIVTIVGLKPETTYAVRLAALNGKGLGEISAASEF'

    binding = BindingAffinity(tfr)
    seq = ["CC[C@H](C)[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)O"]
    
    scores = binding(seq)
    print(scores)
    print(len(scores))

if __name__ == '__main__':
    unittest()