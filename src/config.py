import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description='PepTune Training and Evaluation')
    
    # Noise parameters
    noise_group = parser.add_argument_group('noise')
    noise_group.add_argument('--noise-type', type=str, default='loglinear', 
                            help='Type of noise schedule')
    noise_group.add_argument('--sigma-min', type=float, default=1e-4,
                            help='Minimum sigma value')
    noise_group.add_argument('--sigma-max', type=float, default=20,
                            help='Maximum sigma value')
    noise_group.add_argument('--state-dependent', action='store_true', default=True,
                            help='Use state-dependent noise')
    
    # Base parameters
    parser.add_argument('--base-path', type=str, default='/path/to/PepTune',
                       help='Base path to PepTune')
    parser.add_argument('--mode', type=str, default='ppl_eval',
                       choices=['train', 'ppl_eval', 'sample_eval'],
                       help='Running mode')
    parser.add_argument('--diffusion', type=str, default='absorbing_state',
                       help='Diffusion type')
    parser.add_argument('--vocab', type=str, default='old_smiles',
                       choices=['old_smiles', 'new_smiles', 'selfies', 'helm'],
                       help='Vocabulary type')
    parser.add_argument('--backbone', type=str, default='roformer',
                       choices=['peptideclm', 'helmgpt', 'dit', 'roformer', 'finetune_roformer'],
                       help='Model backbone')
    parser.add_argument('--parameterization', type=str, default='subs',
                       help='Parameterization type')
    parser.add_argument('--time-conditioning', action='store_true', default=False,
                       help='Use time conditioning')
    parser.add_argument('--T', type=int, default=0,
                       help='Number of diffusion steps (0 for continuous time, 1000 for discrete)')
    parser.add_argument('--subs-masking', action='store_true', default=False,
                       help='Use substitution masking')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # MCTS parameters
    mcts_group = parser.add_argument_group('mcts')
    mcts_group.add_argument('--mcts-num-children', type=int, default=50,
                           help='Number of children in MCTS')
    mcts_group.add_argument('--mcts-num-objectives', type=int, default=5,
                           help='Number of objectives in MCTS')
    mcts_group.add_argument('--mcts-topk', type=int, default=100,
                           help='Top-k for MCTS')
    mcts_group.add_argument('--mcts-mask-token', type=int, default=4,
                           help='Mask token ID')
    mcts_group.add_argument('--mcts-num-iter', type=int, default=128,
                           help='Number of MCTS iterations')
    mcts_group.add_argument('--mcts-sampling', type=int, default=0,
                           help='Sampling strategy (0 for gumbel, >0 for top-k)')
    mcts_group.add_argument('--mcts-invalid-penalty', type=float, default=0.5,
                           help='Penalty for invalid sequences')
    mcts_group.add_argument('--mcts-sample-prob', type=float, default=1.0,
                           help='Sampling probability')
    mcts_group.add_argument('--mcts-perm', action='store_true', default=True,
                           help='Use permutation in MCTS')
    mcts_group.add_argument('--mcts-dual', action='store_true', default=False,
                           help='Use dual mode')
    mcts_group.add_argument('--mcts-single', action='store_true', default=False,
                           help='Use single mode')
    mcts_group.add_argument('--mcts-time-dependent', action='store_true', default=True,
                           help='Use time-dependent MCTS')
    
    # Data parameters
    data_group = parser.add_argument_group('data')
    data_group.add_argument('--train-data', type=str,
                           default='/path/to/your/home/PepTune/data/peptide_data',
                           help='Path to training data')
    data_group.add_argument('--valid-data', type=str,
                           default='/path/to/your/home/PepTune/data/peptide_data',
                           help='Path to validation data')
    data_group.add_argument('--data-batching', type=str, default='wrapping',
                           choices=['padding', 'wrapping'],
                           help='Batching strategy')
    
    # Loader parameters
    loader_group = parser.add_argument_group('loader')
    loader_group.add_argument('--global-batch-size', type=int, default=64,
                             help='Global batch size')
    loader_group.add_argument('--eval-global-batch-size', type=int, default=None,
                             help='Evaluation global batch size (defaults to global-batch-size)')
    loader_group.add_argument('--num-workers', type=int, default=None,
                             help='Number of dataloader workers (defaults to available CPUs)')
    loader_group.add_argument('--pin-memory', action='store_true', default=True,
                             help='Pin memory for dataloaders')
    
    # Sampling parameters
    sampling_group = parser.add_argument_group('sampling')
    sampling_group.add_argument('--predictor', type=str, default='ddpm_cache',
                               choices=['analytic', 'ddpm', 'ddpm_cache'],
                               help='Predictor type for sampling')
    sampling_group.add_argument('--num-sequences', type=int, default=100,
                               help='Number of sequences to generate')
    sampling_group.add_argument('--sampling-eps', type=float, default=1e-3,
                               help='Sampling epsilon')
    sampling_group.add_argument('--steps', type=int, default=128,
                               help='Number of sampling steps')
    sampling_group.add_argument('--seq-length', type=int, default=100,
                               help='Sequence length')
    sampling_group.add_argument('--noise-removal', action='store_true', default=True,
                               help='Use noise removal')
    sampling_group.add_argument('--num-sample-batches', type=int, default=2,
                               help='Number of sample batches')
    sampling_group.add_argument('--num-sample-log', type=int, default=2,
                               help='Number of samples to log')
    sampling_group.add_argument('--stride-length', type=int, default=1,
                               help='Stride length for sampling')
    sampling_group.add_argument('--num-strides', type=int, default=1,
                               help='Number of strides')
    
    # Training parameters
    training_group = parser.add_argument_group('training')
    training_group.add_argument('--antithetic-sampling', action='store_true', default=True,
                               help='Use antithetic sampling')
    training_group.add_argument('--training-sampling-eps', type=float, default=1e-3,
                               help='Training sampling epsilon')
    training_group.add_argument('--focus-mask', action='store_true', default=False,
                               help='Use focus mask')
    training_group.add_argument('--accumulator', action='store_true', default=False,
                               help='Use accumulator')
    
    # Evaluation parameters
    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument('--checkpoint-path', type=str, default=None,
                           help='Path to checkpoint for evaluation')
    eval_group.add_argument('--disable-ema', action='store_true', default=False,
                           help='Disable EMA')
    eval_group.add_argument('--compute-generative-perplexity', action='store_true', default=False,
                           help='Compute generative perplexity')
    eval_group.add_argument('--perplexity-batch-size', type=int, default=8,
                           help='Batch size for perplexity computation')
    eval_group.add_argument('--compute-perplexity-on-sanity', action='store_true', default=False,
                           help='Compute perplexity on sanity check')
    eval_group.add_argument('--gen-ppl-eval-model', type=str, default='gpt2-large',
                           help='Model for generative perplexity evaluation')
    eval_group.add_argument('--generate-samples', action='store_true', default=True,
                           help='Generate samples during evaluation')
    eval_group.add_argument('--generation-model', type=str, default=None,
                           help='Model for generation')
    
    # Optimizer parameters
    optim_group = parser.add_argument_group('optim')
    optim_group.add_argument('--weight-decay', type=float, default=0.075,
                            help='Weight decay')
    optim_group.add_argument('--lr', type=float, default=3e-4,
                            help='Learning rate')
    optim_group.add_argument('--beta1', type=float, default=0.9,
                            help='Adam beta1')
    optim_group.add_argument('--beta2', type=float, default=0.999,
                            help='Adam beta2')
    optim_group.add_argument('--eps', type=float, default=1e-8,
                            help='Adam epsilon')
    
    # PepCLM model parameters
    pepclm_group = parser.add_argument_group('pepclm')
    pepclm_group.add_argument('--pepclm-hidden-size', type=int, default=768,
                             help='PepCLM hidden size')
    pepclm_group.add_argument('--pepclm-cond-dim', type=int, default=256,
                             help='PepCLM conditioning dimension')
    pepclm_group.add_argument('--pepclm-n-heads', type=int, default=20,
                             help='PepCLM number of attention heads')
    pepclm_group.add_argument('--pepclm-n-blocks', type=int, default=4,
                             help='PepCLM number of blocks')
    pepclm_group.add_argument('--pepclm-dropout', type=float, default=0.5,
                             help='PepCLM dropout rate')
    pepclm_group.add_argument('--pepclm-length', type=int, default=512,
                             help='PepCLM sequence length')
    
    # General model parameters
    model_group = parser.add_argument_group('model')
    model_group.add_argument('--model-type', type=str, default='ddit',
                            help='Model type')
    model_group.add_argument('--hidden-size', type=int, default=768,
                            help='Model hidden size')
    model_group.add_argument('--cond-dim', type=int, default=128,
                            help='Conditioning dimension')
    model_group.add_argument('--length', type=int, default=512,
                            help='Sequence length')
    model_group.add_argument('--n-blocks', type=int, default=12,
                            help='Number of blocks')
    model_group.add_argument('--n-heads', type=int, default=12,
                            help='Number of attention heads')
    model_group.add_argument('--scale-by-sigma', action='store_true', default=True,
                            help='Scale by sigma')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
    
    # RoFormer parameters
    roformer_group = parser.add_argument_group('roformer')
    roformer_group.add_argument('--roformer-hidden-size', type=int, default=768,
                               help='RoFormer hidden size')
    roformer_group.add_argument('--roformer-n-layers', type=int, default=8,
                               help='RoFormer number of layers')
    roformer_group.add_argument('--roformer-n-heads', type=int, default=8,
                               help='RoFormer number of attention heads')
    roformer_group.add_argument('--roformer-max-position-embeddings', type=int, default=1035,
                               help='RoFormer max position embeddings')
    
    # HelmGPT parameters
    helmgpt_group = parser.add_argument_group('helmgpt')
    helmgpt_group.add_argument('--helmgpt-hidden-size', type=int, default=256,
                              help='HelmGPT hidden size')
    helmgpt_group.add_argument('--helmgpt-embd-pdrop', type=float, default=0.1,
                              help='HelmGPT embedding dropout')
    helmgpt_group.add_argument('--helmgpt-resid-pdrop', type=float, default=0.1,
                              help='HelmGPT residual dropout')
    helmgpt_group.add_argument('--helmgpt-attn-pdrop', type=float, default=0.1,
                              help='HelmGPT attention dropout')
    helmgpt_group.add_argument('--helmgpt-ff-dropout', type=float, default=0.0,
                              help='HelmGPT feedforward dropout')
    helmgpt_group.add_argument('--helmgpt-block-size', type=int, default=140,
                              help='HelmGPT block size')
    helmgpt_group.add_argument('--helmgpt-n-layer', type=int, default=8,
                              help='HelmGPT number of layers')
    helmgpt_group.add_argument('--helmgpt-n-heads', type=int, default=8,
                              help='HelmGPT number of attention heads')
    
    # Trainer parameters
    trainer_group = parser.add_argument_group('trainer')
    trainer_group.add_argument('--accelerator', type=str, default='cuda',
                              help='Accelerator type')
    trainer_group.add_argument('--num-nodes', type=int, default=1,
                              help='Number of nodes')
    trainer_group.add_argument('--devices', type=int, default=1,
                              help='Number of devices')
    trainer_group.add_argument('--gradient-clip-val', type=float, default=1.0,
                              help='Gradient clipping value')
    trainer_group.add_argument('--precision', type=str, default='64-true',
                              help='Training precision')
    trainer_group.add_argument('--num-sanity-val-steps', type=int, default=2,
                              help='Number of sanity validation steps')
    trainer_group.add_argument('--max-epochs', type=int, default=100,
                              help='Maximum number of epochs')
    trainer_group.add_argument('--max-steps', type=int, default=1_000_000,
                              help='Maximum number of steps')
    trainer_group.add_argument('--log-every-n-steps', type=int, default=10,
                              help='Log every n steps')
    trainer_group.add_argument('--limit-train-batches', type=float, default=1.0,
                              help='Limit training batches')
    trainer_group.add_argument('--limit-val-batches', type=float, default=1.0,
                              help='Limit validation batches')
    trainer_group.add_argument('--check-val-every-n-epoch', type=int, default=1,
                              help='Check validation every n epochs')
    
    # WandB parameters
    wandb_group = parser.add_argument_group('wandb')
    wandb_group.add_argument('--wandb-project', type=str, default='peptune',
                            help='WandB project name')
    wandb_group.add_argument('--wandb-notes', type=str, default=None,
                            help='WandB notes')
    wandb_group.add_argument('--wandb-group', type=str, default=None,
                            help='WandB group')
    wandb_group.add_argument('--wandb-job-type', type=str, default=None,
                            help='WandB job type')
    wandb_group.add_argument('--wandb-name', type=str, default='sophia-tang',
                            help='WandB run name')
    wandb_group.add_argument('--wandb-id', type=str, default=None,
                            help='WandB run ID')
    
    # Checkpointing parameters
    checkpoint_group = parser.add_argument_group('checkpointing')
    checkpoint_group.add_argument('--save-dir', type=str, default=None,
                                 help='Directory to save checkpoints')
    checkpoint_group.add_argument('--resume-from-ckpt', action='store_true', default=True,
                                 help='Resume from checkpoint')
    checkpoint_group.add_argument('--resume-ckpt-path', type=str, default=None,
                                 help='Path to checkpoint to resume from')
    checkpoint_group.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                                 help='Save checkpoint every n epochs')
    checkpoint_group.add_argument('--checkpoint-monitor', type=str, default='val/nll',
                                 help='Metric to monitor for checkpointing')
    checkpoint_group.add_argument('--checkpoint-save-top-k', type=int, default=10,
                                 help='Save top k checkpoints')
    checkpoint_group.add_argument('--checkpoint-mode', type=str, default='min',
                                 choices=['min', 'max'],
                                 help='Mode for checkpoint monitoring')
    checkpoint_group.add_argument('--checkpoint-dirpath', type=str,
                                 default='./checkpoints/11M-old-tokenizer',
                                 help='Directory path for checkpoints')
    
    # LR Scheduler parameters
    scheduler_group = parser.add_argument_group('lr_scheduler')
    scheduler_group.add_argument('--lr-warmup-steps', type=int, default=2500,
                                help='Number of warmup steps for learning rate')
    
    return parser


def get_args():
    """Parse and return arguments."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Post-process arguments
    if args.eval_global_batch_size is None:
        args.eval_global_batch_size = args.global_batch_size
    
    if args.num_workers is None:
        args.num_workers = len(os.sched_getaffinity(0))
    
    if args.wandb_id is None:
        args.wandb_id = f"{args.wandb_name}_nov12_set2"
    
    if args.save_dir is None:
        args.save_dir = os.getcwd()
    
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
