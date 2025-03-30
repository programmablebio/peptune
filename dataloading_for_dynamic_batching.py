#!/usr/bin/env
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset,load_from_disk
import sys
import lightning.pytorch as pl
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from functools import partial
import re


class DynamicBatchingDataset(Dataset):
    def __init__(self, dataset_dict, tokenizer):
        print('Initializing dataset...')
        self.dataset_dict = {
            'attention_mask': [torch.tensor(item) for item in dataset_dict['attention_mask']],
            'input_ids': [torch.tensor(item) for item in dataset_dict['input_ids']],
            'labels': dataset_dict['labels']
        }
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {
                'input_ids': self.dataset_dict['input_ids'][idx],
                'attention_mask': self.dataset_dict['attention_mask'][idx],
                'labels': self.dataset_dict['labels'][idx]
            }
        elif isinstance(idx, list):
            return {
                'input_ids': [self.dataset_dict['input_ids'][i] for i in idx],
                'attention_mask': [self.dataset_dict['attention_mask'][i] for i in idx],
                'labels': [self.dataset_dict['labels'][i] for i in idx]
            }   
        else:
            raise ValueError(f"Expected idx to be int or list, but got {type(idx)}")   

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, tokenizer):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        
    def peptide_bond_mask(self, smiles_list):
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
        max_seq_length = 1035 #max(len(smiles) for smiles in smiles_list)  # Find the longest SMILES
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

    def peptide_token_mask(self, smiles_list, token_lists):
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
        atomwise_masks = self.peptide_bond_mask(smiles_list)

    
        for batch_idx, atomwise_mask in enumerate(atomwise_masks):
            token_seq = token_lists[batch_idx]
            atom_idx = 0
            
            for token_idx, token in enumerate(token_seq):
                if token_idx != 0 and token_idx != len(token_seq) - 1:
                    if torch.sum(atomwise_mask[atom_idx:atom_idx+len(token)]) >= 1:
                        tokenized_masks[batch_idx][token_idx] = 1
                    atom_idx += len(token)
        
        return tokenized_masks
    
    def collate_fn(self, batch):
        item = batch[0]
            
        token_array = self.tokenizer.get_token_split(item['input_ids'])
        bond_mask = self.peptide_token_mask(item['labels'], token_array)

        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'bond_mask': bond_mask
        } 

    def train_dataloader(self):
        train_dataset = DynamicBatchingDataset(self.dataset['train'], tokenizer=self.tokenizer)
        return DataLoader(
            train_dataset, 
            batch_size=1, 
            collate_fn=self.collate_fn,  # Use the instance method
            shuffle=True, 
            num_workers=12, 
            pin_memory=True
        )

    def val_dataloader(self):
        val_dataset = DynamicBatchingDataset(self.dataset['val'], tokenizer=self.tokenizer)
        return DataLoader(
            val_dataset, 
            batch_size=1, 
            collate_fn=self.collate_fn,  # Use the instance method
            num_workers=8, 
            pin_memory=True
        )
