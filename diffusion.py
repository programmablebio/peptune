import numpy as np
import sys
import itertools
import time
import torch
from torch import Tensor
import math
import torch.nn.functional as F
import numpy as np
import random as rd
import lightning as L
from torch.distributions.categorical import Categorical
import torchmetrics
from dataclasses import dataclass
import gc
import pickle
import utils.utils as utils

import dataset as dataloader
import models.helmgpt as helmgpt
import models.peptideclm as peptideclm
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
import noise_schedule
from torch.optim.lr_scheduler import _LRScheduler
import models.roformer as roformer 
from utils.filter import PeptideAnalyzer


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  attn_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / math.log(2)


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        
        super().__init__()
        self.config = config
        #self.save_hyperparameters()
        
        # PeptideCLM tokenizer 
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_token_id = self.tokenizer.mask_token_id
        self.sampler = self.config.sampling.predictor
        self.analyzer = PeptideAnalyzer()
        
        # backbone LM PeptideCLM model
        if self.config.backbone == 'peptideclm':
            self.backbone = peptideclm.EncoderWrapper(self.tokenizer)
            self.backbone.unfreeze_all_layers()
            self.backbone = torch.compile(self.backbone)
        elif self.config.backbone == 'helmgpt':
            self.backbone = helmgpt.GPT(self.config, self.tokenizer)
            #self.backbone = torch.compile(self.backbone)
        elif self.config.backbone == 'roformer':
            self.backbone = roformer.Roformer(self.config, self.tokenizer)
            self.backbone.unfreeze_all_layers()
        elif self.config.backbone == 'finetune_roformer':
            self.backbone = roformer.Roformer(self.config, self.tokenizer)
            self.backbone.freeze_model()
            self.backbone.unfreeze_n_layers(n=8)
        else: 
            Exception('invalid backbone config')
        
        self.neg_infinity = -1000000.0
        self.T = config.T
        # noise schedule for non-peptide bond tokens (default to log-linear)
        self.noise = noise_schedule.get_noise(config)
        # noise schedule for peptide bonds (log-polynomial)
        self.bond_noise = noise_schedule.LogPolyNoise()
        self.time_conditioning = self.config.time_conditioning
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        
        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        self.gen_ppl_metric = Perplexity()
                
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='trainer/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        
    """LOSS"""
    
    """LOSS FOR INVALID PEPTIDES"""
    
    @torch.no_grad()
    def conditional_gumbel(self, logits, D, k):
        """
        Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one-hot vector).

        Input:
        - logits: Tensor of shape (batch_size, seq_len, vocab_size)
        - D: One-hot tensor of shape (batch_size, seq_len, vocab_size)
        - k: Number of Gumbel samples

        Output:
        - Adjusted logits with shape (k, batch_size, seq_len, vocab_size)
        """

        # iid. exponential samples of shape (k, batch_size, seq_len, vocab_size)
        E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])

        # E of the chosen class, shape (k, batch_size, seq_len, 1)
        Ei = (D * E).sum(dim=-1, keepdim=True)

        # Partition function (normalization constant), shape (batch_size, seq_len, 1)
        Z = logits.exp().sum(dim=-1, keepdim=True)

        # Adjusted logits for Gumbel distribution
        adjusted = (
            D * (-torch.log(Ei) + torch.log(Z)) +
            (1 - D) * -torch.log(E / logits.exp() + Ei / Z)
        )

        # Adjusted logits shape: (k, batch_size, seq_len, vocab_size)
        return adjusted - logits
        
    def replace_gradient(self, value, surrogate):
        """
        Returns `value` but backpropagates gradients through `surrogate`.
        """
        return surrogate + (value - surrogate).detach()
    
    def gumbel_rao(self, logits, k, temp=1.0, I=None):
        """
        Returns a categorical sample from logits (over axis=-1) as a
        one-hot vector, with gumbel-rao gradient.

        Input:
        - logits: Tensor of shape (batch_size, seq_len, vocab_size)
        - k: Number of Gumbel samples for Rao-Blackwellization
        - temp: Temperature for softmax
        - I: Optional, precomputed categorical sample tensor of shape (batch_size, seq_len)

        Output:
        - One-hot tensor of shape (batch_size, seq_len, vocab_size)
        with Gumbel-Rao gradient.
        """
        assert logits.shape[-1] == self.tokenizer.vocab_size
        vocab_size = logits.shape[-1]

        if I is None:
            # Sample indices for each token in the batch
            I = torch.distributions.categorical.Categorical(logits=logits).sample()  # (batch_size, seq_len)

        # Convert indices to one-hot encodings, shape (batch_size, seq_len, vocab_size)
        D = torch.nn.functional.one_hot(I, num_classes=vocab_size).float()

        # Generate k different adjusted logits that all evaluate to the same sequence
        adjusted = logits + self.conditional_gumbel(logits, D, k=k)  # (k, batch_size, seq_len, vocab_size)

        # Compute the surrogate by averaging softmax across k samples
        surrogate = torch.nn.functional.softmax(adjusted / temp, dim=-1).mean(dim=0)  # (batch_size, seq_len, vocab_size)

        # Return one-hot representation with surrogate gradient
        return self.replace_gradient(D, surrogate)
    
    def compute_invalid_loss(self, logits, k=None, temp=None):
        """
        Penalizes logits that produce invalid sequences using the `is_peptide` function,
        scaling penalties inversely with token probabilities.

        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size].
            k: Number of samples for Gumbel-Rao.
            temp: Temperature for softmax.

        Returns:
            loss: A scalar tensor representing the total loss for invalid sequences.
        """

        #samples = self.gumbel_rao(logits, k=k, temp=temp)  # (batch_size, seq_len, vocab_size)

        # Convert logits to sequences using the tokenizer
        batch_token_ids = logits.argmax(dim=-1).to(self.device)  # (batch_size, seq_len)
        sampled_sequences = self.tokenizer.batch_decode(batch_token_ids)

        # Check validity of each sampled sequence (not differentiable)
        penalties = torch.tensor(
            [1 if not self.analyzer.is_peptide(seq) else 0 for seq in sampled_sequences],
            dtype=torch.float32,
            device=self.device
        )  
        #print(penalties)

        # Compute probabilities for each token (batch_size, seq_length)
        sampled_probs = torch.softmax(logits, dim=-1).gather(dim=-1, index=batch_token_ids.unsqueeze(-1)).squeeze(-1).to(self.device)

        # scale penalties by softmax probability of sampled tokens
        scaled_penalty = penalties[:, None] * sampled_probs # (batch_size, seq_length)
        
        return scaled_penalty.to(self.device)
    
    """DIFFUSION LOSS"""
    
    def sample_t(self, n, device):
        """
            Sample random time steps for batch training
        """
        # sample values uniformly at random from [0, 1)
        eps_t = torch.rand(n, device=device) 
        # antithetic sampling: reduce variance by pairing each sample with complementary sample
        if self.config.training.antithetic_sampling:
            # compute interval between sampled time steps
            offset = torch.arange(n, device=device) / n
            # ensure that each eps value is evenly spaced between [0, 1)
            eps_t = ((eps_t / n) + offset) % 1

        # ensures values are not exactly 0 or 1
        t = (1 - self.config.training.sampling_eps) * eps_t + self.config.training.sampling_eps
        
        return t
        
    """def mask_samples(self, x0, mask_prob):
        
        # generate array of values in range [0, 1] uniformly at random
        # will be used to determine which tokens are masked
        mask_indices = torch.rand(* x0.shape, device=x0.device) # (batch_size, L)
        
        # select tokens to mask if the random value in mask_indices is less than mask_prob
        # this will mask approximately the fraction of tokens indicated by mask_prob
        zt = torch.where(mask_indices < mask_prob, self.mask_token_id, x0)
        
        return zt"""
    
    def q_xt(self, x, mask_prob):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """

        actual_seq_length = (x != 0).sum(dim=-1, keepdim=True)
        #print(actual_seq_length)

        max_mask_length = (actual_seq_length * 0.75).long()

        mask_indices = torch.rand(*x.shape, device=x.device) < mask_prob
        
        restricted_move_indices = torch.zeros_like(mask_indices, dtype=torch.bool)

        for i in range(x.shape[0]):
            true_positions = torch.where(mask_indices[i])[0]
            if len(true_positions) > max_mask_length[i]:
                selected_positions = true_positions[:max_mask_length[i].item()]
                restricted_move_indices[i, selected_positions] = True
            else:
                restricted_move_indices[i] = mask_indices[i]
                
        xt = torch.where(restricted_move_indices, self.tokenizer.mask_token_id, x)

        return xt

    
    def sample_prior(self, *batch_dims):
        """
            Returns array of fully masked sequences with same shape as input
        """
        return self.mask_token_id * torch.ones(* batch_dims, dtype=torch.int64)
    

    """COMPUTING LOSS"""
    
    def compute_diffusion_loss(self, model_output, xt, x0, t):
        """
        Computes diffusion loss term in ELBO 
        (evaluates how accurately the model predicts the token probabilities at each time step)
        
        Inputs:
        - model_output: [sequence length, vocab size, vocab size] array of logits for each token at each sequence position
        - zt: corrupted version of original input x0 at timestep t
        - x0: original input sequence
        - t: timestep
        """
        # compute interval between each timestep
        dt = 1 / self.T
        
        # compute vectorized alpha scaling terms for the logits at timestep s and t
        alpha_t = 1 - t + torch.zeros_like(x0)
        # s = t - dt
        alpha_s = 1 - (t - dt) + torch.zeros_like(x0)
        
        # gather vector of log-probabilities for each token in x0
        # log<x_theta, x>
        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]) # shape (B, L, vocab_size)
        # gather log-probabillities for assigning a masked token at each position in the sequence at time t 
        # log<x_theta, m>
        log_x_theta_at_m = model_output[:, :, self.mask_token_id]
        # obtain non-log probability of assigning a masked token
        # <xt, m>
        x_theta_at_m = log_x_theta_at_m.exp()
        
        # first term of diffusion loss
        term_1_coef = dt / t
        term_1_log_numerator = torch.log((alpha_t * x_theta_at_m) / t + 1)
        term_1_log_denom = log_x_theta_at_x0
        
        # second term of diffusion loss
        term_2_coef = 1 - (dt / t)
        term_2_log_numerator = term_1_log_numerator
        term_2_log_denom = torch.log((alpha_s * x_theta_at_m) / (t - dt) + 1)
        
        L_vb_masked = (term_1_coef * (term_1_log_numerator - term_1_log_denom) + 
                       term_2_coef * (term_2_log_numerator - term_2_log_denom))
        
        # multiply by <zt, m> term
        L_vb = L_vb_masked * (xt == self.mask_token_id)
        
        # scale by T and return
        return self.T * L_vb
    
    """def _forward_pass_diffusion(self, x0, attn_mask, mask=None):
        
        print(x0)
        # randomly sample time steps to start the denoising process for each x0 in batch
        t = self.sample_t(x0.shape[0], x0.device)
        
        # if we are training the intermediate transition blocks
        if self.T > 0: 
            # scale by total timesteps T and cast to integer
            t = (t * self.T).to(torch.int)
            # scale down by T to get a multiple of 1/T
            t = t / self.T
            # add 1/T to ensure no 0 values
            t += (1 / self.T)
        
        # get noise and rate of noise at timestep t
        sigma, dsigma = self.noise(t)
        time_conditioning = sigma[:, None]
        # get masking probabilities for all tokens for each batch
        mask_prob = 1 - torch.exp(-sigma[:, None]) # (batch_size, L)
        
        # get masked samples at different timesteps
        if mask is None: zt = self.q_xt(x0, mask_prob)
        else: zt = x0.where(mask==1, torch.full_like(x0, self.mask_token_id))
        
        model_output = self.forward(zt, attn_mask, time_conditioning) 
        
        utils.print_nans(model_output, 'model_output')
        
        if self.T > 0:
            # compute diffusion loss
            diffusion_loss = self.compute_diffusion_loss(model_output, zt, x0, t)
            return diffusion_loss
        
        # compute loss for the final that converts from z0 to x0
        # -log(p_theta)
        # get (batch_size, L) array of log-probabilities 
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1) # (B, L)
        
        
        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]"""
    
    def _forward_pass_diffusion(self, x0, attn_mask, bond_mask=None, mask=None):
        """
            Training reverse diffusion model x_theta to reconstruct samples x0
            
            bond_mask: (batch, seq_length)
        """
        # randomly sample time steps to start the denoising process for each x0 in batch
        t = self.sample_t(x0.shape[0], self.device)
        
        # if we are training the intermediate transition blocks
        if self.T > 0: 
            # scale by total timesteps T and cast to integer
            t = (t * self.T).to(torch.int)
            # scale down by T to get a multiple of 1/T
            t = t / self.T
            # add 1/T to ensure no 0 values
            t += (1 / self.T)
        
        # get noise and rate of noise at timestep t
        # sigma = -log(1-t); dsigma = 1 / (1-t)
        sigma, dsigma = self.noise(t)
        time_conditioning = sigma[:, None]
        
        # Get masking probabilities for all tokens for each batch
        # log-linear: 1 - alpha = t
        base_mask_prob = 1 - torch.exp(-sigma[:, None])  # (batch_size, L)

        if self.config.noise.state_dependent and (bond_mask is not None):
            # log-polynomial masking schedule: alpha = 1 - t^w
            # bond_sigma = -log(1-t^w) for w = 3 (default)
            # bond_dsigma = -wt^(w-1) / (1-t^w)
            bond_sigma, bond_dsigma = self.bond_noise(t) # scalar
            # expand dimensions for broadcasting to (B, L)
            bond_sigma = bond_sigma[:, None] 
            bond_dsigma = bond_dsigma[:, None]
            sigma = sigma[:, None]
            dsigma = dsigma[:, None]
            
            # compute masking probability for peptide bonds 1 - bond_alpha = t^w
            bond_mask_prob = 1 - torch.exp(-bond_sigma).to(self.device)
            # piece together (B, L) tensor with modified masking prob at peptide-bond locations
            mask_prob = torch.where(bond_mask == 1, bond_mask_prob, base_mask_prob).to(self.device)
            #print(mask_prob)
            dsigma = torch.where(bond_mask == 1, bond_dsigma, dsigma).to(self.device)
            sigma = torch.where(bond_mask == 1, bond_sigma, sigma).to(self.device)
        else:
            mask_prob = base_mask_prob.to(self.device)
        
        # get masked samples at different timesteps
        if mask is None: 
            zt = self.q_xt(x0, mask_prob).to(self.device)
        else: 
            zt = x0.where(mask==1, torch.full_like(x0, self.mask_token_id)).to(self.device)
        
        model_output = self.forward(zt, attn_mask=attn_mask.to(self.device), sigma=time_conditioning).to(self.device)
        
        # debugging
        assert not torch.isnan(model_output).any()
        assert model_output.is_cuda
        utils.print_nans(model_output, 'model_output')
        
        # compute invalid loss
        invalid_loss = self.compute_invalid_loss(logits=model_output).to(self.device) # (B, L)
        #print(invalid_loss)
        
        if self.T > 0:
            # compute diffusion loss
            diffusion_loss = self.compute_diffusion_loss(model_output, zt, x0, t)
            return diffusion_loss
        
        # compute loss for the final that converts from z0 to x0
        # -log(p_theta)
        # get (batch_size, L) array of log-probabilities 
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1).to(self.device) # (B, L)
        
        if self.config.noise.state_dependent and (bond_mask is not None):
            return (-log_p_theta * (dsigma / torch.expm1(sigma)) + invalid_loss).to(self.device)
        else: 
            return ((-log_p_theta * (dsigma / torch.expm1(sigma))[:, None]) + invalid_loss).to(self.device)

    def _loss(self, x0, attn_mask, bond_mask=None, mask=None):        
        loss = self._forward_pass_diffusion(x0, attn_mask, bond_mask, mask)
        
        # negative log loss
        nlls = loss * attn_mask
            
        # count number of tokens
        num_tokens = attn_mask.sum()
        
        # compute batch loss
        batch_nll = nlls.sum()
        # compute per token loss
        token_nll = batch_nll / num_tokens
        # return losses
        return Loss(loss = token_nll.to(self.device), nlls = nlls.to(self.device), attn_mask = attn_mask.to(self.device))
    
    def _compute_loss(self, batch, prefix, bond_mask=None):
        
        attn_mask = batch['attention_mask'].to(self.device)
            
        if 'mask' in batch: 
            mask = batch['mask'].to(self.device)
        else: 
            mask = None
        
        if 'bond_mask' in batch:
            bond_mask = batch['bond_mask'].to(self.device)
        else:
            bond_mask = None
        
        losses = self._loss(batch['input_ids'].to(self.device), attn_mask, bond_mask, mask)
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(
                losses.nlls.to(self.device), 
                losses.attn_mask.to(self.device)
            )
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(
                losses.nlls.to(self.device), 
                losses.attn_mask.to(self.device)
            )
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.attn_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')
        
        self.log_dict(metrics,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
        
        return loss
        
    
    """SAMPLING"""
    
    def generate_from_masked(self, num_samples=None, seq_length=None, sample_steps=128, eps=1e-5):
        # get number of timesteps
        if sample_steps is None:
            sample_steps = self.config.sampling.steps
        
        if seq_length is None:
            seq_length = self.config.sampling.seq_length

        # sample fully masked sequences
        z = self.sample_prior(num_samples, seq_length).to(self.device)
        
        # create vector of sample_steps timesteps
        timesteps = torch.linspace(1, eps, sample_steps + 1, device=self.device)
        
        # compute interval between timesteps
        dt = (1 - eps) / sample_steps
        
        for i in range(sample_steps):
            t = timesteps[i] * torch.ones(z.shape[0], 1, device=self.device)
            
            z = self.single_reverse_step(z, t, dt)
        
        return z
    
    
    """SAMPLING STEP"""
    
    def single_reverse_step(self, zt, t, dt, attn_mask=None):
        """
            Take a single reverse diffusion step for the expansion step of the MCTS algorithm
        """
        # get sigma values that determine masking prob
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        
        # reshape sigmas
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        
        # compute masking probabilities for each timestep
        change_prob_t = 1 - torch.exp(-sigma_t)
        change_prob_s = 1 - torch.exp(-sigma_s)
        
        # expand dimensions
        change_prob_t = change_prob_t[:, None, None]
        change_prob_s = change_prob_s[:, None, None]
        
        # get prodiction model that outputs token probabilities
        log_p_x0 = self.forward(zt, attn_mask=attn_mask, sigma=sigma_t) 
        
        # check dimensions match
        assert change_prob_t.ndim == log_p_x0.ndim
        
        # compute reverse diffusion probability of being unmasked at timestep s
        # (sigma_s - sigma_t)*x_theta
        q_zs = log_p_x0.exp() * (change_prob_t - change_prob_s)
        
        # compute reverse diffusion probability of remaining masked at timestep s
        # (1 - sigma_s)*m
        q_zs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        # sample sequence at timestep s from categorical distribution of q_zs
        z_changed = sample_categorical(q_zs)
        
        copy_flag = (zt != self.mask_token_id).to(zt.dtype)
        return (copy_flag * zt) + ((1 - copy_flag) * z_changed)

    def cached_reverse_step(self, x, t, dt, p_x0=None, attn_mask=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            p_x0 = self.forward(x, attn_mask=attn_mask, sigma=sigma_t).exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        x_changed = sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_token_id).to(x.dtype)
        
        return p_x0, copy_flag * x + (1 - copy_flag) * x_changed
    
    # first step in expansion
    def batch_cached_reverse_step(self, token_array, t, dt, batch_size, p_x0=None, attn_mask=None):
        """
        Generates batch_size different samples from the same starting point for the 
        first expansion step of MCTS

        Args:
            x (_type_): _description_
            t (_type_): _description_
            dt (_type_): _description_
            batch_size (_type_): _description_
            p_x0 (_type_, optional): _description_. Defaults to None.
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if token_array.dim() == 1:
            token_array = token_array.unsqueeze(0)
            #token_array = token_array.repeat(batch_size, 1)
            
        attn_mask = torch.ones_like(token_array)
        
        if p_x0 is None:
            p_x0 = self.forward(token_array, attn_mask=attn_mask, sigma=sigma_t).exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        # repeat the parent token along the first dimension which will be unmasked into distinct sequences
        token_array = token_array.repeat(batch_size, 1)
        
        if self.config.mcts.sampling == 0:
            x_changed = sample_batched_categorical(q_xs.to(self.device), batch_size)
        else:
            x_changed = sample_batched_top_k(q_xs.to(self.device), batch_size, self.config.mcts.sampling)
        
        copy_flag = (token_array != self.mask_token_id).to(token_array.dtype)
        
        return p_x0, copy_flag * token_array + (1 - copy_flag) * x_changed
    
    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma
    
    def forward(self, zt, attn_mask, sigma):
        """
        Predicts the token log-probabilities from zt at time t with noise schedule sigma
        """
        sigma = self._process_sigma(sigma)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32, cache_enabled=True):
            logits = self.backbone(zt, attn_mask).to(self.device)
            
        return self.subs_parameterization(logits, zt)
    
    def subs_parameterization(self, logits, zt):
        """
        Updates reverse diffusion logits based on SUBS parameterization:
        - zero masking probabilities: -infinity probability of being masked during reverse diffusion
        - carry-over unmasking: unmasked input tokens remain unchanged during reverse diffusion

        Args:
            logits: vector of token probabilities for unmasking masked tokens
            zt: partially unmasked sequence at current timestep
        """
        logits[:, :, self.mask_token_id] += self.neg_infinity # [sequence index, current token, next token]
        
        
        logits = (logits - torch.logsumexp(logits, dim=-1, keepdim=True)).to(self.device)
        
        
        unmasked_indices = (zt != self.mask_token_id).to(self.device)  # shape: [200, seq_length]
        batch_idx, seq_idx = torch.where(unmasked_indices)  # Get explicit indices
        batch_idx = batch_idx.to(self.device)
        seq_idx = seq_idx.to(self.device)
        tokens = zt[batch_idx, seq_idx].to(self.device)  # Get the tokens at those positions
        
        assert logits.is_contiguous(), "logits tensor is not contiguous"
        assert unmasked_indices.shape == zt.shape, "same shape"
        assert not torch.isnan(logits).any(), "NaN values found in logits"
        assert tokens.max() < logits.shape[-1], "token indices out of bounds"
        assert batch_idx.max() < logits.shape[0], "batch index out of bounds"
        assert seq_idx.max() < logits.shape[1], "seq index out of bounds"
        assert batch_idx.device == seq_idx.device == logits.device == tokens.device, "device inconsistent"

        logits[batch_idx, seq_idx] = self.neg_infinity  # Set everything to -inf first
        logits[batch_idx, seq_idx, tokens] = 0  # Set only the specific token positions to 0
        # return logits with SUBS parameterization
        return logits.to(self.device)
    
    """SAMPLING"""
    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, x_input=None):
        """ 
            Generate samples
        """
        batch_size_per_gpu = self.config.eval.perplexity_batch_size
        
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        if x_input is not None:
            x = x_input['input_ids'].to(self.device)
            attn_mask = x_input['attention_mask'].to(self.device)
        else:
            x = self.sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
            attn_mask = torch.ones_like(x).to(self.device)
        
        
        timesteps = torch.linspace(1, eps, num_steps+1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        generation_history = [] # used to track which tokens are unmasked
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device = self.device)
            if self.sampler == 'ddpm':
                x = self.single_reverse_step(x, t, dt).to(self.device)
                
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self.cached_reverse_step(x, t, dt, p_x0=p_x0_cache, attn_mask=attn_mask)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next.to(self.device)
                #print(self.tokenizer.decode(x.squeeze()))
            else:
                x = self._analytic_update(x, t, dt, attn_mask).to(self.device)
        
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t).to(self.device)
            else:
                time_conditioning = self.noise(t)[0].to(self.device)
                x = self.forward(x, attn_mask=attn_mask, sigma=time_conditioning).argmax(dim=-1).to(self.device)
                #print(self.tokenizer.decode(x.squeeze()))
        return x.to(self.device)


    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        self.backbone.train()
        self.noise.train()
        return samples

    def get_score(self, zt, sigma, attn_mask=None):
        
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)
        
        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x 
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        
        model_output = self.forward(zt, attn_mask=attn_mask, sigma=sigma)
    
        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1
        
        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_token_id] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1,
            zt[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        
        unmasked_score[:, :, self.mask_token_id] = - (log_k[:, None] * torch.ones_like(zt))
        
        masked_indices = (zt == self.mask_token_id).to(model_output.dtype)[:, :, None]
        
        model_output = (masked_score * masked_indices + unmasked_score * (1 - masked_indices))
        
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_token_id] += extra_const
        return score

    def _analytic_update(self, x, t, step_size, attn_mask=None):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, attn_mask, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_token_id] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
        i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_token_id,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge   
        
    
    """TRAINING from https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py"""
    
    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        self.backbone.train()
        self.noise.train()
    
    
    def training_step(self, batch, batch_idx):
        # Initialize throughput calculation
        start_time = time.time()

        if self.config.vocab == 'old_smiles' or self.config.vocab == 'new_smiles':
            loss = self._compute_loss(batch, prefix='train', bond_mask=batch['bond_mask'])
        else:
            loss = self._compute_loss(batch, prefix='train')
            
        self.log(name='trainer/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True)
        
        # Calculate throughput
        elapsed_time = time.time() - start_time
        total_tokens = batch['input_ids'].numel()
        throughput = total_tokens / elapsed_time

        self.log(name='trainer/throughput',
                value=throughput,
                on_step=True,
                on_epoch=False,
                sync_dist=True)

        return loss
    

    def on_load_checkpoint(self, checkpoint):
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
    
    """VALIDATION"""
    def on_validation_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        if self.config.vocab == 'old_smiles' or self.config.vocab == 'new_smiles':
            loss = self._compute_loss(batch, prefix='val', bond_mask=batch['bond_mask'])
        else:
            loss = self._compute_loss(batch, prefix='val')
            
        self.log(name='trainer/val_loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    """OPTIMIZATION"""

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(),self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )
            
        self.total_steps = self.config.trainer.max_steps
        scheduler = CosineWarmup(optimizer,
                                warmup_steps=self.config.lr_scheduler.num_warmup_steps,
                                total_steps=self.total_steps)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val/loss',
            'name': 'trainer/lr'
        }

        return [optimizer], [scheduler_dict]

    @torch.no_grad()
    def compute_masked_perplexity(self, generated_ids, input_ids):
        """
            Computes masked perplexity between array of generated token ids and masked ids that are converted to logits
        """
        
        total_nll = 0
        total_tokens = 0
        
        input_ids = torch.tensor(input_ids).to(self.device)
        #print(input_ids)

        for sequence in generated_ids:
            # tokenize the sequence
            
            gt_ids = torch.tensor(sequence).to(self.device)
            #print(gt_ids)

            sys.stdout.flush()

            # forward pass thorugh backbone peptideclm model
            attn_mask = torch.ones_like(input_ids).to(self.device)
            
            # compute logits using backbone
            
            if self.config.mode in ['train', 'ppl_eval']:
                outputs = self.backbone.forward(input_ids=input_ids, attn_mask=attn_mask)
            elif self.config.mode == 'sample_eval':
                outputs = self.backbone.forward(input_ids=input_ids)
            
            
            # get logits for each position in sequence across all tokens in vocab
            #logits = outputs[-1] # (batch_size, seq_length, vocab_size)

            logits = outputs.view(-1, outputs.size(-1))  
            gt_ids = gt_ids.view(-1)    
            
            #print(logits.shape)
            #print(gt_ids.shape)

            # compute loss
            # shift_logits = logits[:, :-1, :].contiguous() # remove eos
            # shift_labels = input_ids[:, 1:].contiguous()
            # print(masked)
            
            loss = F.cross_entropy(logits, 
                                    gt_ids.where(input_ids==self.mask_token_id, torch.full_like(gt_ids, -100)).view(-1), 
                                    reduction='sum')

            total_nll += loss.item()
            # count all non-padding tokens
            total_tokens += input_ids.ne(self.tokenizer.pad_token_id).sum().item() # count in bos and eos
            
        # compute pseudo-perplexity
        # print(total_nll, ",;,", total_tokens)
        pseudo_perplexity = torch.exp(torch.tensor(total_nll / total_tokens))
        self.gen_ppl_metric.update(pseudo_perplexity)

        return pseudo_perplexity.item()
    

def sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def sample_batched_categorical(categorical_probs, batch_size):
    """
    Generates `m` distinct sequences sampled from categorical probabilities 
    using the Gumbel distribution to ensure randomness while following probabilities
    
    Args:
        categorical_probs (torch.Tensor): tensor of shape (sequence_length, vocab_length)
                                          representing categorical probabilities
        m (int): number of distinct sequences to sample
    
    Returns:
        torch.Tensor: tensor of shape (m, sequence_length), where each row is a 
                      distinct sequence of sampled category indices.
    """
    _, sequence_length, vocab_size = categorical_probs.shape

    # add Gumbel noise and sample m sequences
    gumbel_noise = (-torch.log(-torch.log(torch.rand(batch_size, sequence_length, vocab_size) + 1e-10) + 1e-10)).to(categorical_probs.device)
    noisy_scores = torch.log(categorical_probs) + gumbel_noise  # add Gumbel noise to log probabilities
    
    # select the highest score (most likely category after Gumbel noise)
    sampled_sequences = noisy_scores.argmax(dim=-1)  # shape: (m, sequence_length)

    return sampled_sequences

def sample_batched_top_k(categorical_probs, batch_size, k):
    """
    Generates `m` sequences sampled from the top-k probabilities of each token
    using Gumbel noise to ensure randomness and reduce bias towards the most likely options.

    Args:
        categorical_probs (torch.Tensor): A tensor of shape (sequence_length, vocab_length)
                                          representing categorical probabilities.
        m (int): Number of sequences to sample.
        k (int): Number of top probabilities to consider for sampling.

    Returns:
        torch.Tensor: A tensor of shape (m, sequence_length), where each row is a 
                      sampled sequence of category indices.
    """
    _, sequence_length, vocab_length = categorical_probs.shape

    # Add Gumbel noise to the log probabilities
    gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, sequence_length, vocab_length) + 1e-10) + 1e-10).to(categorical_probs.device)
    noisy_scores = torch.log(categorical_probs[None, :, :]) + gumbel_noise  # Shape: (m, sequence_length, vocab_length)

    # Get the top-k categories based on noisy scores
    top_k_scores, top_k_indices = torch.topk(noisy_scores, k, dim=-1)  # Shape: (m, sequence_length, k)

    # Convert top-k scores back to probabilities and normalize
    top_k_probs = torch.softmax(top_k_scores, dim=-1).to(categorical_probs.device)  # Shape: (m, sequence_length, k)

    # Sample randomly from the top-k probabilities
    sampled_indices_in_top_k = torch.multinomial(top_k_probs.reshape(-1, k), num_samples=1).squeeze(-1).to(categorical_probs.device)
    sampled_indices_in_top_k = sampled_indices_in_top_k.view(batch_size, sequence_length).to(categorical_probs.device)  # Shape: (batch_size, sequence_length)

    # Map sampled indices back to the original vocabulary indices
    sampled_sequences = torch.gather(top_k_indices, -1, sampled_indices_in_top_k.unsqueeze(-1)).squeeze(-1).to(categorical_probs.device)

    return sampled_sequences

def unsqueeze(x, reference):
    return x.view(* x.shape, * ((1,) * (len(reference.shape) - len(x.shape))))

class CosineWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]