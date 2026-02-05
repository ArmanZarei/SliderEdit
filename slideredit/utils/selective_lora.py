from contextlib import contextmanager
from collections import OrderedDict
import torch
import torch.nn as nn
from slideredit.models import SelectiveLoRALinear


def flux_kontext_find_substring_token_indices(prompt: str, substr: str, tokenizer):
    prompt_tokens = tokenizer(prompt).input_ids

    substr_tokens = tokenizer(substr).input_ids[:-1]

    start_idx = -1
    for i in range(len(prompt_tokens) - len(substr_tokens) + 1):
        if prompt_tokens[i:i+len(substr_tokens)] == substr_tokens:
            start_idx = i
            break   

    assert start_idx != -1, "substr_tokens not found in tokens"

    token_indices = list(range(start_idx, start_idx + len(substr_tokens)))

    if tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr:
        print("============================ Warning ============================")
        print("[Warning] tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr")
        print(f"[Warning] Decoded: {tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices])}")
        print(f"[Warning] Expected: {substr}")
        print("=================================================================")

    return token_indices


def create_stlora_tokens_mask_and_scale(
    prompt: str,
    subprompts: list[str],
    control_weights: list[float],
    tokenizer,
    token_indices_finder_fn,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
):
    """
    Create token mask and LoRA scale tensors for selective LoRA application.
    
    Args:
        prompt: The full prompt text
        subprompts: List of substring prompts to apply selective LoRA to
        control_weights: List of weight values for each subprompt (same length as subprompts)
        tokenizer: Tokenizer used to find token positions
        token_indices_finder_fn: Function to find token indices for substrings
        device: Device to create tensors on
        dtype: Data type for the scale tensor
        
    Returns:
        tuple: (tokens_mask, lora_scale) where both are tensors of shape (1, max_seq_length)
    """
    assert len(subprompts) == len(control_weights), "subprompts and control_weights must have same length"
    
    max_seq_length = tokenizer.model_max_length
    tokens_mask = torch.zeros((1, max_seq_length), device=device, dtype=torch.bool)
    lora_scale = torch.ones((1, max_seq_length), device=device, dtype=dtype)
    
    for subprompt, weight in zip(subprompts, control_weights):
        token_indices = token_indices_finder_fn(prompt, subprompt, tokenizer)
        tokens_mask[0, token_indices] = True
        lora_scale[0, token_indices] *= weight

    return tokens_mask, lora_scale


def set_stlora_scale(transformer, scale):
    for m in transformer.modules():
        if isinstance(m, SelectiveLoRALinear):
            m.set_scaling(scale)


@contextmanager
def stlora_token_mask_ctx(model, mask: torch.Tensor, disable_mask_after=True):  # mask: (B, S) bool
    """
    Context manager to set token mask for all SelectiveLoRALinear modules in the model.

    Args:
        model: The model containing SelectiveLoRALinear modules.
        mask: A boolean tensor of shape (B, S) indicating which tokens to apply LoRA to.
        disable_mask_after: If True, the token mask will be cleared after exiting the context.
    """
    loras = [m for m in model.modules() if isinstance(m, SelectiveLoRALinear)]
    for m in loras:
        m.set_token_mask(mask)
    try:
        yield
    finally:
        if disable_mask_after:
            for m in loras:
                m.set_token_mask(None)


def _get_parent(model, dotted_name):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_selective_lora_modules(model, target_linear_name_suffixes, r, alpha, dropout):
    replaced = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SelectiveLoRALinear":  # Already loaded (Useful for debugging)
            module = module.base

        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            parent, attr = _get_parent(model, name)
            wrapped = SelectiveLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            replaced.append(name)

    return replaced


def delete_selective_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SelectiveLoRALinear):
            parent, attr = _get_parent(model, name)
            setattr(parent, attr, module.base)


def save_selective_lora_state_dict(unwraped_model, output_file_path):
    ckpt = OrderedDict()
    for name, module in unwraped_model.named_modules():
        if isinstance(module, SelectiveLoRALinear):
            ckpt[f"{name}.lora_A.weight"] = module.lora_A.weight.cpu()
            ckpt[f"{name}.lora_B.weight"] = module.lora_B.weight.cpu()

    torch.save(ckpt, output_file_path)
