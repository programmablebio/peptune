import torch
import math
import sys
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer


def mask_for_de_novo(config, sequence_length):
    if config.vocab == 'helm':
        return "[MASK]" * sequence_length
    elif config.vocab == 'new_smiles' or config.vocab == 'selfies':
        return ["<mask>"] * sequence_length
    else: 
        return ["[MASK]"] * sequence_length

def generate_de_novo(sequence_length, tokenizer, model):
    masked_sequence = mask_for_de_novo(sequence_length)
    inputs = tokenizer(masked_sequence, return_tensors='pt').to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    logits_at_masks = logits[0, mask_token_indices]

    pred_tokens = []
    for i in mask_token_indices:
        topk_logits, topk_indices = logits_at_masks[i].topk(k=3, dim=-1)
        probabilities = torch.nn.functional.softmax(topk_logits, dim=-1)
        predicted_index = torch.distributions.categorical.Categorical(probabilities).sample()
        predicted_token_id = topk_indices[predicted_index].item()
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
        pred_tokens.append(predicted_token)
    
    generated_sequence = ''.join(pred_tokens)
    perplexity = calculate_perplexity(model, tokenizer, generated_sequence)

    return (generated_sequence, perplexity)


def calculate_perplexity(model, tokenizer, generated_sequence, mask_token_indices):
    total_loss = 0.0
    tensor_input = tokenizer.encode(generated_sequence, return_tensors='pt').to(model.device)

    for i in mask_token_indices:
        masked_input = tensor_input.clone()
        masked_input[0, i] = tokenizer.mask_token_id
    
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()
    
    num_mask_tokens = len(mask_token_indices)
    if num_mask_tokens == 0:
        perplexity = 10000
    else:
        avg_loss = total_loss / num_mask_tokens
        perplexity = math.exp(avg_loss)

    return perplexity


def calculate_cosine_sim(original_sequence, generated_sequence, tokenizer, pepclm_model, device):
    og_embeddings = pepclm_model.roformer.encoder(original_sequence)
    new_embeddings = pepclm_model.roformer.encoder(generated_sequence)

    sequence_similarity = torch.nn.functional.cosine_similarity(og_embeddings, new_embeddings, dim=-1)
    cosine_similarity = torch.mean(sequence_similarity).item()
    return cosine_similarity
    

def calculate_hamming_dist(original_sequence, generated_sequence):
    generated_sequence = generated_sequence
    original_sequence = original_sequence
    return sum(1 if original_sequence[i] != generated_sequence[i] else 0 for i in range(len(original_sequence)))