
import re
import torch

import utils
    
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from functools import partial
import sys

class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = int(self.indices[idx])
        item = self.dataset[actual_idx]
        return item


# for weighting losses of peptide bonds
def peptide_bond_mask(smiles_list):
    """
    Returns a mask with shape (batch_size, seq_length) that has 1 at the locations
    of recognized bonds in the positions dictionary and 0 elsewhere.

    Args:
        smiles_list: List of peptide SMILES strings (batch of SMILES strings).

    Returns:
        np.ndarray: A mask of shape (batch_size, seq_length) with 1s at bond positions.
    """
    # Initialize the batch mask
    batch_size = len(smiles_list)
    max_seq_length = max(len(smiles) for smiles in smiles_list)  # Find the longest SMILES
    mask = torch.zeros((batch_size, max_seq_length), dtype=torch.int)  # Mask filled with zeros

    bond_patterns = [
        (r'OC\(=O\)', 'ester'),
        (r'N\(C\)C\(=O\)', 'n_methyl'),
        (r'N[12]C\(=O\)', 'peptide'),  # Pro peptide bonds
        (r'NC\(=O\)', 'peptide'),  # Regular peptide bonds
        (r'C\(=O\)N\(C\)', 'n_methyl'),
        (r'C\(=O\)N[12]?', 'peptide')
    ]

    for batch_idx, smiles in enumerate(smiles_list):
        positions = []
        used = set()

        # Identify bonds
        for pattern, bond_type in bond_patterns:
            for match in re.finditer(pattern, smiles):
                if not any(p in range(match.start(), match.end()) for p in used):
                    positions.append({
                        'start': match.start(),
                        'end': match.end(),
                        'type': bond_type,
                        'pattern': match.group()
                    })
                    used.update(range(match.start(), match.end()))

        # Update the mask for the current SMILES
        for pos in positions:
            mask[batch_idx, pos['start']:pos['end']] = 1

    return mask

def peptide_token_mask(smiles_list, token_lists):
    """
    Returns a mask with shape (batch_size, num_tokens) that has 1 for tokens
    where any part of the token overlaps with a peptide bond, and 0 elsewhere.

    Args:
        smiles_list: List of peptide SMILES strings (batch of SMILES strings).
        token_lists: List of tokenized SMILES strings (split into tokens).

    Returns:
        np.ndarray: A mask of shape (batch_size, num_tokens) with 1s for peptide bond tokens.
    """
    # Initialize the batch mask
    batch_size = len(smiles_list)
    token_seq_length = max(len(tokens) for tokens in token_lists)  # Find the longest tokenized sequence
    tokenized_masks = torch.zeros((batch_size, token_seq_length), dtype=torch.int)  # Mask filled with zeros
    atomwise_masks = peptide_bond_mask(smiles_list)

 
    for batch_idx, atomwise_mask in enumerate(atomwise_masks):
        token_seq = token_lists[batch_idx]
        atom_idx = 0
        
        for token_idx, token in enumerate(token_seq):
            if token_idx != 0 and token_idx != len(token_seq) - 1:
                if torch.sum(atomwise_mask[atom_idx:atom_idx+len(token)]) >= 1:
                    tokenized_masks[batch_idx][token_idx] = 1
                atom_idx += len(token)
    
    return tokenized_masks

def extract_amino_acid_sequence(helm_string):
    """
    Extracts the amino acid sequence from a HELM peptide notation and outputs it as an array,
    removing any brackets around each amino acid.

    Args:
        helm_string (str): The HELM notation string for a peptide.

    Returns:
        list: A list containing each amino acid in sequence without brackets.
    """
    # Use regex to find the pattern within `{}` brackets following "PEPTIDE" followed by a number
    matches = re.findall(r'PEPTIDE\d+\{([^}]+)\}', helm_string)
    
    if matches:
        # Join all matched sequences and split by dots to get individual amino acids
        amino_acid_sequence = []
        for match in matches:
            sequence = match.replace('[', '').replace(']', '').split('.')
            amino_acid_sequence.extend(sequence)
        return amino_acid_sequence
    else:
        return "Invalid HELM notation or no peptide sequence found."
    
def helm_collate_fn(batch, tokenizer):
    sequences = [item['HELM'] for item in batch]
    
    max_len = 0
    for sequence in sequences:
        seq_len = len(extract_amino_acid_sequence(sequence))
        if seq_len > max_len:
            max_len = seq_len
    
    tokens = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask']
    }
    
    
def collate_fn(batch, tokenizer):
    """Standard data collator that truncates/pad sequences based on max_length"""
    valid_sequences = []
    valid_items = []
    
    for item in batch: 
        try:
            test_tokens = tokenizer([item['SMILES']], return_tensors='pt', padding=False, truncation=True, max_length=1035)
            valid_sequences.append(item['SMILES'])
            valid_items.append(item)
        except Exception as e:
            print(f"Skipping sequence due to: {str(e)}")
            continue
    
    #sequences = [item['SMILES'] for item in batch]
    #max_len = max([len(seq) for seq in sequences])
    #labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)

    tokens = tokenizer(valid_sequences, return_tensors='pt', padding=True, truncation=True, max_length=1035)
    
    token_array = tokenizer.get_token_split(tokens['input_ids'])
    bond_mask = peptide_token_mask(valid_sequences, token_array)
    #attention_masks = torch.ones(tokens.size()[:2], dtype=torch.bool)

    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'bond_mask': bond_mask
    }
        

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, tokenizer, batch_size, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        #self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
                          num_workers=8, 
                          pin_memory=True
                          )
    

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
                          num_workers=8, 
                          pin_memory=True
                          )
  
    """def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
                          num_workers=8, pin_memory=True)"""