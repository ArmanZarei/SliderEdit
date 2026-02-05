import os
from enum import Enum
import torch
from diffusers import FluxKontextPipeline
from slideredit.utils import inject_selective_lora_modules, flux_kontext_find_substring_token_indices, set_stlora_scale, create_stlora_tokens_mask_and_scale, stlora_token_mask_ctx, delete_selective_lora_modules
from peft.tuners.lora.layer import Linear as PeftLoRALinear
from slideredit.utils.selective_lora import _get_parent


LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


class LoRAAdapterType(Enum):
    NONE = "none"
    GSTLORA = "gstlora"
    STLORA = "stlora"


class SliderEditFluxKontextPipeline(FluxKontextPipeline):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pipe = super().from_pretrained(*args, **kwargs)
        pipe.__class__ = cls
        pipe.loaded_adapter = LoRAAdapterType.NONE
        return pipe
    
    def unload_slideredit_lora(self):
        if self.loaded_adapter.value == LoRAAdapterType.NONE.value:
            return

        print("Unloading previously loaded adapter...")
        if self.loaded_adapter.value == LoRAAdapterType.GSTLORA.value:
            self.transformer.delete_adapters(["gstlora"])
            # Remove PeftLoRALinear wrappers
            for name, module in self.transformer.named_modules():
                if isinstance(module, PeftLoRALinear):
                    parent, attr = _get_parent(self.transformer, name)
                    setattr(parent, attr, module.base_layer)
        elif self.loaded_adapter.value == LoRAAdapterType.STLORA.value:
            delete_selective_lora_modules(self.transformer)
        self.loaded_adapter = LoRAAdapterType.NONE

    def load_gstlora(self, ckpt: str):
        self.unload_slideredit_lora()
        self.load_lora_weights(ckpt, weight_name="pytorch_lora_weights.safetensors", adapter_name="gstlora")
        self.transformer.to(self.device)
        self.loaded_adapter = LoRAAdapterType.GSTLORA
    
    def load_stlora(self, ckpt: str, lora_rank: int, lora_dropout: float):
        self.unload_slideredit_lora()
        replaced = inject_selective_lora_modules(self.transformer, LORA_TARGET_MODULES, r=lora_rank, alpha=lora_rank, dropout=lora_dropout)
        missing, unexpected = self.transformer.load_state_dict(torch.load(ckpt), strict=False)
        assert len(unexpected) == 0, f"Unexpected keys when loading STLora weights: {unexpected}"
        self.transformer.to(self.device)
        self.loaded_adapter = LoRAAdapterType.STLORA
    

    def _gstlora_forward(self, slider_alpha: float, **kwargs):
        assert slider_alpha is not None, "slider_alpha must be provided for GSTLoRA."
        self.transformer.set_adapters(["gstlora"], [slider_alpha])
        return super().__call__(**kwargs)

    def _stlora_forward(self, prompt: str, subprompts_list: list[str], slider_alpha_list: list[float], **kwargs):
        assert len(subprompts_list) == len(slider_alpha_list), "subprompts_list and slider_alpha_list must have the same length."

        tokens_mask, lora_scale = create_stlora_tokens_mask_and_scale(
            prompt=prompt,
            subprompts=subprompts_list,
            control_weights=slider_alpha_list,
            tokenizer=self.tokenizer_2,
            token_indices_finder_fn=flux_kontext_find_substring_token_indices,
            device=self.device,
            dtype=torch.bfloat16
        )
        set_stlora_scale(self.transformer, lora_scale.unsqueeze(-1))

        with stlora_token_mask_ctx(self.transformer, tokens_mask):
            out = super().__call__(prompt=prompt, **kwargs)
        
        return out

    def __call__(self, **kwargs):
        if self.loaded_adapter.value == LoRAAdapterType.NONE.value:
            return super().__call__(**kwargs)
        elif self.loaded_adapter.value == LoRAAdapterType.GSTLORA.value:
            return self._gstlora_forward(**kwargs)
        elif self.loaded_adapter.value == LoRAAdapterType.STLORA.value:
            return self._stlora_forward(**kwargs)
        else:
            raise ValueError(f"Unknown loaded_adapter: {self.loaded_adapter}")

