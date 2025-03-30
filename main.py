#!/usr/bin/env
import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
import uuid

import wandb
import fsspec
import hydra
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import omegaconf
import rich.syntax
import rich.tree
import torch
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append("/home/st512/peptune/scripts/peptide-mdlm-mcts")

import dataset as dataloader
import dataloading_for_dynamic_batching as dynamic_dataloader
from diffusion import Diffusion
import utils.utils as utils
from new_tokenizer.ape_tokenizer import APETokenizer

from lightning.pytorch.strategies import DDPStrategy
from datasets import load_dataset
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from helm_tokenizer.helm_tokenizer import HelmTokenizer


#wandb.login(key="5a7613c531cb58f9802f3f8e2f73bc4997b917ab")

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def _load_from_checkpoint(config, tokenizer):
	if 'hf' in config.backbone:
		return Diffusion(
			config, tokenizer=tokenizer).to('cuda')
	else:
		model = Diffusion.load_from_checkpoint(
			config.eval.checkpoint_path,
			tokenizer=tokenizer,
			config=config)

	return model

@L.pytorch.utilities.rank_zero_only
def print_config(
	config: omegaconf.DictConfig,
	resolve: bool = True,
	save_cfg: bool = True) -> None:
	"""
 	Prints content of DictConfig using Rich library and its tree structure.
	
	Args:
		config (DictConfig): Configuration composed by Hydra.
		resolve (bool): Whether to resolve reference fields of DictConfig.
		save_cfg (bool): Whether to save the configuration tree to a file.
	"""

	style = 'dim'
	tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

	fields = config.keys()
	for field in fields:
		branch = tree.add(field, style=style, guide_style=style)

		config_section = config.get(field)
		branch_content = str(config_section)
		if isinstance(config_section, omegaconf.DictConfig):
			branch_content = omegaconf.OmegaConf.to_yaml(
			config_section, resolve=resolve)

		branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
	rich.print(tree)
	if save_cfg:
		with fsspec.open(
			'{}/config_tree.txt'.format(
			config.checkpointing.save_dir), 'w') as fp:
			rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def print_batch(train_ds, valid_ds, tokenizer, k=64):
  	#for dl_type, dl in [
    #('train', train_ds), ('valid', valid_ds)]:
    
	for dl_type, dl in [
		('train', train_ds)]:
		print(f'Printing {dl_type} dataloader batch.')
		batch = next(iter(dl))
		print('Batch input_ids.shape', batch['input_ids'].shape)
		first = batch['input_ids'][0, :k]
		last = batch['input_ids'][0, -k:]
		print(f'First {k} tokens:', tokenizer.decode(first))
		print('ids:', first)
		print(f'Last {k} tokens:', tokenizer.decode(last))
		print('ids:', last)


def generate_samples(config, logger, tokenizer):
	logger.info('Generating samples.')
	model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
	# model.gen_ppl_metric.reset()
	
	#stride_length = config.sampling.stride_length
	#num_strides = config.sampling.num_strides
 
	for _ in range(config.sampling.num_sample_batches):
		samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
		peptide_sequences = model.tokenizer.batch_decode(samples)
		model.compute_generative_perplexity(peptide_sequences)
  
	print('Peptide samples:', peptide_sequences)
 
	print('Generative perplexity:', model.compute_masked_perplexity())
  
	return peptide_sequences


def ppl_eval(config, logger, tokenizer, data_module):
	logger.info('Starting Zero Shot Eval.')

	model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

	wandb_logger = None
	if config.get('wandb', None) is not None:
		wandb_logger = L.pytorch.loggers.WandbLogger(
		config=omegaconf.OmegaConf.to_object(config),
		** config.wandb)
  
	callbacks = []
 
	if 'callbacks' in config:
		for _, callback in config.callbacks.items():
			callbacks.append(hydra.utils.instantiate(callback))
   
	trainer = hydra.utils.instantiate(
		config.trainer,
		default_root_dir=os.getcwd(),
		callbacks=callbacks,
		strategy=DDPStrategy(find_unused_parameters = True),
		logger=wandb_logger)
  
	#_, valid_ds = dataloader.get_dataloaders(config, tokenizer, skiptrain=True, valid_seed=config.seed)
	trainer.test(model, data_module)


def _train(config, logger, tokenizer, data_module):
	logger.info('Starting Training.')
	wandb_logger = None

	if config.get('wandb', None) is not None:
		unique_id = str(uuid.uuid4())

		config.wandb.id = f"{config.wandb.id}_{unique_id}"

		wandb_logger = L.pytorch.loggers.WandbLogger(
			config=omegaconf.OmegaConf.to_object(config),
			** config.wandb)

	if (config.checkpointing.resume_from_ckpt
		and config.checkpointing.resume_ckpt_path is not None
		and utils.fsspec_exists(
			config.checkpointing.resume_ckpt_path)):
		ckpt_path = config.checkpointing.resume_ckpt_path
	else:
		ckpt_path = None

	# Lightning callbacks
	callbacks = []
	if 'callbacks' in config:
		for callback_name, callback_config in config.callbacks.items():
			if callback_name == 'model_checkpoint':
				model_checkpoint_config = {k: v for k, v in callback_config.items() if k != '_target_'}
				callbacks.append(ModelCheckpoint(**model_checkpoint_config))
			else:
				callbacks.append(hydra.utils.instantiate(callback_config))
    
	if config.training.accumulator:
		accumulator = GradientAccumulationScheduler(scheduling = {1: 5, 2: 4, 3: 3, 4: 1})
		callbacks.append(accumulator)
  
	trainer = hydra.utils.instantiate(
		config.trainer,
		default_root_dir=os.getcwd(),
		callbacks=callbacks,
		accelerator='cuda',
		strategy=DDPStrategy(find_unused_parameters = True),
		devices=[2,3,4,5,6,7],
		logger=wandb_logger)
	
	model = Diffusion(config, tokenizer=tokenizer)
	
	if config.backbone == 'finetune_roformer':
		checkpoint = torch.load('/home/st512/peptune/scripts/peptide-mdlm-mcts/checkpoints/11M-old-tokenizer/epoch=1-step=24080.ckpt')
		model.load_state_dict(checkpoint['state_dict'])
	
	trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

  
@hydra.main(version_base=None, config_path='/home/st512/peptune/scripts/peptide-mdlm-mcts', config_name='config')
def main(config):
	"""
 		Main entry point for training
   """   
	wandb.init(project="peptune")
	L.seed_everything(config.seed)
 
	# print_config(config, resolve=True, save_cfg=True)

	logger = utils.get_logger(__name__)
	# load PeptideCLM tokenizer
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

	if config.backbone == 'finetune_roformer':
		train_dataset = load_dataset('csv', data_files=config.data.train)
		val_dataset = load_dataset('csv', data_files=config.data.valid)

		train_dataset = train_dataset['train']#.select(lst)
		val_dataset = val_dataset['train']#.select(lst)
		data_module = dataloader.CustomDataModule(train_dataset, val_dataset, None, tokenizer, batch_size=config.loader.global_batch_size)
	else:
		data_module = dynamic_dataloader.CustomDataModule('/home/st512/peptune/scripts/peptide-mdlm-mcts/data/smiles/11M_smiles_old_tokenizer_no_limit', tokenizer)
	
	if config.mode == 'sample_eval':
		generate_samples(config, logger, tokenizer)
	elif config.mode == 'ppl_eval':
		ppl_eval(config, logger, tokenizer, data_module)
	else:
		_train(config, logger, tokenizer, data_module)


if __name__ == '__main__':
	main()