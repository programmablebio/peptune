import sys
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
import os
from multiprocessing import Pool
from tqdm import tqdm
import lightning.pytorch as pl
sys.path.append('/home/yz927/projects/peptune/scripts/')
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
global_tokenizer = None


def init_pool(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer

class SequenceDataset:
    def __init__(self, sequences, tokenizer, max_sequence_length, num_cores=8):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.num_cores = 8
        self.tokenized_sequences = []
        self.original_sequences = []

    def tokenize_sequences(self):
        print(f"Starting parallel tokenization using {self.num_cores} cores")
        with Pool(processes=self.num_cores, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            results = list(tqdm(
                pool.imap(standalone_tokenize_function, self.sequences),
                total=len(self.sequences)
            ))

        for result, seq in zip(results, self.sequences):
            if result is not None and len(result['input_ids'][0]) <= self.max_sequence_length:
                self.tokenized_sequences.append(result)
                self.original_sequences.append(seq)

    
    def process_sequences(self, batch_size):
        self.tokenize_sequences()
        
        lengths = [(len(seq['input_ids'][0]), i) for i, seq in enumerate(self.tokenized_sequences)]
        lengths.sort()
        
        batches = []
        sequence_batches = []
        current_batch = []
        current_sequence_batch = []
        current_length = 0
        
        for length, idx in tqdm(lengths):
            if current_length + length > self.max_sequence_length or len(current_batch) == batch_size:
                if current_batch:
                    batches.append([self.tokenized_sequences[i] for i in current_batch])
                    sequence_batches.append([self.original_sequences[i] for i in current_batch])
                current_batch = [idx]
                current_sequence_batch = [self.original_sequences[idx]]
                current_length = length
            else:
                current_batch.append(idx)
                current_sequence_batch.append(self.original_sequences[idx])
                current_length += length
                
        if current_batch:
            batches.append([self.tokenized_sequences[i] for i in current_batch])
            sequence_batches.append([self.original_sequences[i] for i in current_batch])
        
        token_batch_fn = TokenizeBatch(self.tokenizer)
        processed_batches = [token_batch_fn(batch) for batch in tqdm(batches)]
        
        dataset = Dataset.from_dict({
            'attention_mask': [batch['attention_mask'] for batch in processed_batches],
            'input_ids': [batch['input_ids'] for batch in processed_batches],
            'labels': sequence_batches
        })
        
        return dataset

class DynamicBatchingDataset(Dataset):
    """
    Process dynamically batched datasets of Huggingface Datasets object. Need special handling since in the previous
    steps, each batch (row in the Datasets object) is already processed for per batch loading
    """

    def __init__(self, dataset_dict):
        print('Initializing dataset...')
        self.dataset_dict = {
            'attention_mask': [torch.tensor(item) for item in dataset_dict['attention_mask']],
            'input_ids': [torch.tensor(item) for item in dataset_dict['input_ids']],
            'labels': dataset_dict['labels']  # Store original sequences as it is
        }

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {
                'attention_mask': self.dataset_dict['attention_mask'][idx],
                'input_ids': self.dataset_dict['input_ids'][idx],
                'labels': self.dataset_dict['labels'][idx]
            }
        elif isinstance(idx, list):
            return {
                'attention_mask': [self.dataset_dict['attention_mask'][i] for i in idx],
                'input_ids': [self.dataset_dict['input_ids'][i] for i in idx],
                'labels': [self.dataset_dict['labels'][i] for i in idx]
            }   
        else:
            raise ValueError(f"Expected idx to be int or list, but got {type(idx)}")    

    @staticmethod
    def collate_fn(batch, verbose=False):
        item = batch[0]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels']
        }

def standalone_tokenize_function(sequence):
    global global_tokenizer
    try:
        tokens = global_tokenizer(sequence)
        # The tokenizer already returns lists of integers, so we just need to wrap them in another list
        # to match the expected format [batch_size, sequence_length]
        return {
            'input_ids': [tokens['input_ids']],
            'attention_mask': [tokens['attention_mask']]
        }
    except Exception as e:
        print(f"Error tokenizing sequence '{sequence}': {e}")
        return None
    
class TokenizeBatch:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batches):
        data_tokens = [torch.tensor(batch['input_ids'][0]) for batch in batches]
        data_tokens_padded = torch.nn.utils.rnn.pad_sequence(data_tokens, batch_first=True, padding_value=self.pad_token_id)
        attention_masks = (data_tokens_padded != self.pad_token_id).long()
        
        return {
            'input_ids': data_tokens_padded,
            'attention_mask': attention_masks,
        }

class PretrainSequenceDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 input_dataset_path,
                 output_dataset_path,
                 num_workers,
                 batch_size,
                 max_sequence_length=512,):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_path = input_dataset_path
        self.output_path = output_dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
    def prepare_data(self):
        if not os.path.exists(self.output_path):
            print("Loading text files")
            with open(f"{self.input_path}/train.txt", 'r') as f:
                train_sequences = [line.strip() for line in f if line.strip()]
            with open(f"{self.input_path}/val.txt", 'r') as f:
                val_sequences = [line.strip() for line in f if line.strip()]
            
            print("Processing training data")
            train_dataset = SequenceDataset(train_sequences, 
                                        self.tokenizer,
                                        self.max_sequence_length)
            print("Processing validation data")
            val_dataset = SequenceDataset(val_sequences,
                                        self.tokenizer,
                                        self.max_sequence_length)

            processed_train = train_dataset.process_sequences(self.batch_size)
            processed_val = val_dataset.process_sequences(self.batch_size)
            
            print("Combining datasets")
            combined_dataset = DatasetDict({
                'train': processed_train,
                'val': processed_val,
            })
            
            print(f"Saving dataset to {self.output_path}")
            combined_dataset.save_to_disk(self.output_path)
        
    def setup(self, stage: str):
        print("Loading processed dataset")
        dataset = load_from_disk(self.output_path)
        self.train_dataset = DynamicBatchingDataset(dataset['train'])
        self.val_dataset = DynamicBatchingDataset(dataset['val'])
    
    def train_dataloader(self):
        print("Creating training dataloader")
        return DataLoader(self.train_dataset, 
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=DynamicBatchingDataset.collate_fn,
                        pin_memory=True)
    
    def val_dataloader(self):
        print("Creating validation dataloader")
        return DataLoader(self.val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=DynamicBatchingDataset.collate_fn,
                        pin_memory=True)
        

if __name__ == '__main__':
    tokenizer = SMILES_SPE_Tokenizer('/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_vocab.txt', 
                                 '/home/st512/peptune/scripts/peptide-mdlm-mcts/tokenizer/new_splits.txt')
    dm = PretrainSequenceDataModule(
        tokenizer=tokenizer,
        input_dataset_path='/home/yz927/projects/peptune/tokens/11M_smiles',
        output_dataset_path='/home/yz927/projects/peptune/tokenized/11M_smiles_old_tokenizer_no_limit',
        num_workers=8,
        batch_size=2000,
        max_sequence_length=16*1000,
    )
    dm.prepare_data()
    dm.setup('fit')
    dm.train_dataloader()
    dm.val_dataloader()
