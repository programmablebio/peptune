from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from tqdm import tqdm
import selfies as sf
from multiprocessing import Pool, cpu_count
from functools import partial
def generate_fingerprint_batch_selfies(selfies_batch):
    fps = []
    valid_selfies = []
    
    for selfies in tqdm(selfies_batch, desc="Generating fingerprints", leave=False):
        try:
            # Convert SELFIES to SMILES then to molecule
            smiles = sf.decoder(selfies)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                valid_selfies.append(selfies)
        except:
            continue
            
    return np.array(fps), valid_selfies

def process_batch(batch, n_clusters, seed):
    fps, valid_selfies = generate_fingerprint_batch_selfies(batch)
    if len(fps) > 0:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
        clusterer.fit(fps)
        labels = clusterer.predict(fps)
        return list(zip(labels, valid_selfies))
    return []

def parallel_clustering_split_selfies(selfies_list, batch_size=10000, n_clusters=1000, train_ratio=0.9, seed=42):
    np.random.seed(seed)
    
    # Create batches
    batches = [selfies_list[i:i + batch_size] 
               for i in range(0, len(selfies_list), batch_size)]
    
    # Initialize parallel processing
    n_cores = 12
    process_batch_partial = partial(process_batch, n_clusters=n_clusters, seed=seed)
    
    cluster_assignments = defaultdict(list)
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_batch_partial, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Combine results
    for batch_results in results:
        for label, selfies in batch_results:
            cluster_assignments[label].append(selfies)
    
    # Split into train/val
    clusters = list(cluster_assignments.values())
    np.random.shuffle(clusters)
    
    train_selfies = []
    val_selfies = []
    total_mols = sum(len(cluster) for cluster in clusters)
    
    for cluster in tqdm(clusters, desc="Splitting clusters"):
        if len(train_selfies) / total_mols < train_ratio:
            train_selfies.extend(cluster)
        else:
            val_selfies.extend(cluster)
    
    print(f"Final splits: Train={len(train_selfies)}, Validation={len(val_selfies)}")
    return train_selfies, val_selfies

try:
    with open('/home/yz927/projects/peptune/tokens/filtered_peptides_selfies.txt', 'r') as f:
        selfies_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(selfies_list)} selfies sequences from file")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the file at file")
except Exception as e:
    raise Exception(f"Error reading file: {str(e)}")

train_selfies, val_selfies = parallel_clustering_split_selfies(
    selfies_list,
    batch_size=10000,
    n_clusters=1000,
    train_ratio=0.8
)
with open('/home/yz927/projects/peptune/tokens/11M_selfies/train_selfies.txt', 'w') as f:
    for line in train_selfies:
        f.write(f"{line}\n")
with open('/home/yz927/projects/peptune/tokens/11M_selfies/val_selfies.txt', 'w') as f:
    for line in val_selfies:
        f.write(f"{line}\n")