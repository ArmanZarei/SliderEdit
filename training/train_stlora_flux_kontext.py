import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import FluxTransformer2DModel
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
import math
from diffusers import FluxKontextPipeline
from tqdm.auto import tqdm
import wandb
import os
from contextlib import nullcontext
from omegaconf import OmegaConf
import torch.nn as nn
import math
from slideredit.models.selective_lora import SelectiveLoRALinear
from slideredit.utils import flux_kontext_find_substring_token_indices, inject_selective_lora_modules, stlora_token_mask_ctx, save_selective_lora_state_dict
from datasets import load_dataset
from accelerate.utils import DataLoaderConfiguration
from accelerate.utils import set_seed
import itertools
import random
from diffusers.utils import make_image_grid

from slideredit.pipelines import SliderEditFluxKontextPipeline, LoRAAdapterType


class CustomMultiEditInstructionDataset(Dataset):
    def __init__(self, images_dir, single_edit_prompts, max_num_instructions_per_prompt):
        self.single_prompts = single_edit_prompts

        self.max_num_instructions_per_prompt = max_num_instructions_per_prompt

        self.prompt_combs = []
        for num_instructions in range(1, max_num_instructions_per_prompt + 1):
            for prompt_combination in itertools.permutations(self.single_prompts, num_instructions):
                self.prompt_combs.append(prompt_combination)

        self.images = []
        for p in Path(images_dir).glob("*"):
            if p.suffix not in [".jpg", ".jpeg", ".png"]:
                continue
            self.images.append(Image.open(p))

        print(f"[Custom Multi-EditInstruction Dataset] Found {len(self.images)} images.")


    def __len__(self):
        return len(self.images) * len(self.prompt_combs)

    def __getitem__(self, index):
        image_index = index // len(self.prompt_combs)
        prompt_comb_index = index % len(self.prompt_combs)

        prompt_comb = self.prompt_combs[prompt_comb_index]

        number_of_instructions_to_supress = random.randint(1, len(prompt_comb))
        indices_to_supress = random.sample(range(len(prompt_comb)), number_of_instructions_to_supress)

        return {
            "orig_edit_prompt": " and ".join(prompt_comb),
            "instructions_to_apply_prompt": " and ".join([prompt_comb[i] for i in range(len(prompt_comb)) if i not in indices_to_supress]),
            "instructions_to_suppress": [prompt_comb[i] for i in indices_to_supress],
            "image": self.images[image_index],
        }


def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "orig_edit_prompts": [item["orig_edit_prompt"] for item in batch],
        "instructions_to_apply_prompts": [item["instructions_to_apply_prompt"] for item in batch],
        "instructions_to_suppress": [item["instructions_to_suppress"] for item in batch],
    }


def load_models(args):
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    t5_tokenizer = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    clip_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    t5_text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    return clip_tokenizer, t5_tokenizer, clip_text_encoder, t5_text_encoder, vae, transformer, noise_scheduler


def encode_prompt_with_clip(tokenizer, text_encoder, prompts, device):
    text_input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids

    pooled_prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=text_encoder.dtype, device=device) # TODO: dtype and device needed ? 
    
    return pooled_prompt_embeds


def encode_prompt_with_t5(tokenizer, text_encoder, prompts, device):
    text_input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).input_ids
    
    prompt_embeds = text_encoder(text_input_ids.to(device))[0].to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds


@torch.no_grad()
def compute_text_embeddings(prompts, t5_tokenizer, t5_text_encoder, clip_tokenizer, clip_text_encoder):
    pooled_prompt_embeds = encode_prompt_with_clip(clip_tokenizer, clip_text_encoder, prompts, device=clip_text_encoder.device)
    prompt_embeds = encode_prompt_with_t5(t5_tokenizer, t5_text_encoder, prompts, device=t5_text_encoder.device)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=t5_text_encoder.device, dtype=t5_text_encoder.dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma


@torch.no_grad()
def get_gt_model_pred(transformer, model_input, timesteps, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids, orig_inp_shape):
    with stlora_token_mask_ctx(transformer, None):
        gt_model_pred = transformer(
            hidden_states=model_input,
            timestep=timesteps / 1000,  # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        gt_model_pred = gt_model_pred[:, :orig_inp_shape[1]]

    return gt_model_pred


@torch.no_grad()
def prepare_latents(pipe, images):
    latents_arr, image_latents_arr, latent_ids_arr, image_ids_arr = [], [], [], []
    for image in images:
        image = pipe.image_processor.preprocess(image, 1024, 1024)
        latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
            image,
            1,
            pipe.transformer.config.in_channels // 4,
            1024,
            1024,
            torch.bfloat16,
            pipe._execution_device,
            None
        )
        latent_ids, image_ids = latent_ids.unsqueeze(0), image_ids.unsqueeze(0)

        latents_arr.append(latents)
        image_latents_arr.append(image_latents)
        latent_ids_arr.append(latent_ids)
        image_ids_arr.append(image_ids)

    latents, image_latents, latent_ids, image_ids = torch.cat(latents_arr, dim=0), torch.cat(image_latents_arr, dim=0), torch.cat(latent_ids_arr, dim=0), torch.cat(image_ids_arr, dim=0)

    return latents, image_latents, latent_ids, image_ids


def get_save_model_hook(accelerator):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                print("Type of model in save_model_hook:", type(model))
                if isinstance(accelerator.unwrap_model(model), FluxTransformer2DModel):
                    save_selective_lora_state_dict(accelerator.unwrap_model(model), os.path.join(output_dir, f"selective_lora.pt"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                
                weights.pop() # make sure to pop weight so that corresponding model is not saved again
    
    return save_model_hook


def get_load_model_hook(accelerator):
    def load_model_hook(models, input_dir):
        assert len(models) == 1, "Only transformer should be passed to the load hook"
        transformer = models.pop()
        missing, unexpected = transformer.load_state_dict(torch.load(os.path.join(input_dir, "selective_lora.pt"), map_location=accelerator.device), strict=False)
        if len(unexpected) > 0:
            raise Exception(f"Unexpected keys found when loading the model: {unexpected}.")
        cast_training_params([transformer], dtype=torch.float32)
        print(f"[Loading checkpoint Successful] Loaded LoRA weights from {os.path.join(input_dir, 'selective_lora.pt')}, {len(missing)} missing keys.")
    
    return load_model_hook


def log_validation(pipeline: SliderEditFluxKontextPipeline, args, accelerator):
    print("Running validation...")

    images_arr = []
    captions = []
    with nullcontext():
        for i, prompt_pair in tqdm(enumerate(args.validation_prompts)):
            prompt = " and ".join(prompt_pair)
            captions.append(prompt)
            images_arr.append([])
            validation_image = Image.open(args.validation_images[i])
            for lora_scale_0, lora_scale_1 in itertools.product(args.validation_lora_scales, repeat=2):
                images_arr[-1].append(pipeline(
                    image=validation_image,
                    prompt=prompt,
                    width=1024,
                    height=1024,
                    generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                    subprompts_list=prompt_pair,
                    slider_alpha_list=[lora_scale_0, lora_scale_1],
                ).images[0].resize((512, 512)))
            captions.append(prompt)
    
        for m in pipeline.transformer.modules():
            if isinstance(m, SelectiveLoRALinear):
                m.reset_scaling()
    
    accelerator.trackers[0].log({
        "validation": [
            wandb.Image(
                make_image_grid(images, rows=len(args.validation_lora_scales), cols=len(args.validation_lora_scales)),
                caption=prompt
            )
            for images, prompt in zip(images_arr, captions)
        ],
    })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(print_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )

    args = OmegaConf.load(parser.parse_args().config)

    if print_args:
        print("="*40, "Arguments", "="*40)
        for arg in args:
            print(f"{arg}: {getattr(args, arg)}")
        print("="*(91))

    return args


def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=dataloader_config,
    )

    set_seed(args.seed)

    clip_tokenizer, t5_tokenizer, clip_text_encoder, t5_text_encoder, vae, transformer, noise_scheduler = load_models(args)

    # Only train the LoRA layers
    for m in [clip_text_encoder, t5_text_encoder, vae, transformer]:
        m.requires_grad_(False)
    
    weight_dtype = torch.bfloat16
    for m in [clip_text_encoder, t5_text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)
    
    transformer.enable_gradient_checkpointing()

    LORA_TARGET_MODULES = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    # for i in range(len(transformer.single_transformer_blocks)):
    #     LORA_TARGET_MODULES.append(f"single_transformer_blocks.{i}.attn.to_q")
    #     LORA_TARGET_MODULES.append(f"single_transformer_blocks.{i}.attn.to_k")
    #     LORA_TARGET_MODULES.append(f"single_transformer_blocks.{i}.attn.to_v")
    #     LORA_TARGET_MODULES.append(f"single_transformer_blocks.{i}.proj_mlp")
    #     LORA_TARGET_MODULES.append(f"single_transformer_blocks.{i}.proj_out")
    replaced = inject_selective_lora_modules(transformer, LORA_TARGET_MODULES, r=args.lora_rank, alpha=args.lora_rank, dropout=args.lora_dropout)
    # for m in transformer.single_transformer_blocks.modules():
    #     if isinstance(m, SelectiveLoRALinear):
    #         m.apply_zero_padding = True

    accelerator.register_save_state_pre_hook(get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(get_load_model_hook(accelerator))

    # Make sure the trainable params are in float32. [only upcast trainable parameters (LoRA) into fp32]
    cast_training_params([transformer], dtype=torch.float32) # TODO: Check if its correct

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW( # TODO: Change to prodigy
        [{"params": transformer_lora_parameters, "lr": args.learning_rate}],
        betas=(0.9, 0.999),
        weight_decay=1e-04,
        eps=1e-08,
    )

    train_dataset = CustomMultiEditInstructionDataset(args.images_dataset_path, args.training_single_edit_prompts, args.max_num_instructions_per_prompt)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True) # TODO: num of workers

    lr_scheduler = get_scheduler(
        "constant", # TODO: make it argument
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    len_train_dataset = len(train_dataset)
    len_train_dataloader = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=OmegaConf.to_container(args, resolve=True), init_kwargs={"wandb":{"name": args.wandb_run_name}})
    
    print("***** Running training *****")
    print(f"  Num examples = {len_train_dataset}")
    print(f"  Num batches each epoch = {len_train_dataloader}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        # Get latest checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None   

        if path is None:
            print(f"No checkpoint found in {args.output_dir}. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path)) 
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = SliderEditFluxKontextPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=clip_text_encoder,
        text_encoder_2=t5_text_encoder,
        transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
        torch_dtype=weight_dtype,
    )
    pipeline.loaded_adapter = LoRAAdapterType.STLORA
    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                images = batch["images"]
                orig_edit_prompts = batch["orig_edit_prompts"]
                instructions_to_apply_prompts = batch["instructions_to_apply_prompts"]
                instructions_to_suppress = batch["instructions_to_suppress"]

                latents, image_latents, latent_ids, image_ids = prepare_latents(pipeline, images)
                latent_ids = torch.cat([latent_ids, image_ids], dim=1) # TODO: Check

                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(orig_edit_prompts, t5_tokenizer, t5_text_encoder, clip_tokenizer, clip_text_encoder)
                neutral_prompt_embeds, neutral_pooled_prompt_embeds, neutral_text_ids = compute_text_embeddings(instructions_to_apply_prompts, t5_tokenizer, t5_text_encoder, clip_tokenizer, clip_text_encoder)

                noise = latents
                bsz = latents.shape[0]

                # Sample a random timestep for each image for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=bsz) # TODO: make it argument with logit_mean, logit_std, mode_scale
                indices = (u * (args.max_train_timesteps - args.min_train_timesteps) + args.min_train_timesteps).long() # instead of noise_scheduler.config.num_train_timesteps
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(noise_scheduler, timesteps, device=latents.device, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None 

                orig_inp_shape = noisy_model_input.shape
                latent_model_input = torch.cat([noisy_model_input, image_latents], dim=1)

                gt_model_pred_neutral = get_gt_model_pred(transformer, latent_model_input, timesteps, guidance, neutral_pooled_prompt_embeds, neutral_prompt_embeds, neutral_text_ids, latent_ids, orig_inp_shape)

                # Set scales
                for m in pipeline.transformer.modules():
                    if isinstance(m, SelectiveLoRALinear):
                        m.reset_scaling()

                # Set mask
                tokens_mask = torch.zeros((bsz, 512), device=accelerator.device, dtype=torch.bool)
                for i in range(bsz):
                    for instruction_to_suppress in instructions_to_suppress[i]:
                        tokens_mask[i, flux_kontext_find_substring_token_indices(orig_edit_prompts[i], instruction_to_suppress, t5_tokenizer)] = True
                with stlora_token_mask_ctx(transformer, tokens_mask, disable_mask_after=False): # This is because of the gradient checkpointing:
                    model_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timesteps / 1000, # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                    model_pred = model_pred[:, :orig_inp_shape[1]]

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas) # TODO: make it argument (same as the one used in compute_density_for_timestep_sampling)
                target = gt_model_pred_neutral # instead of (noise - model_input)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        pipeline.transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
                        log_validation(pipeline, args, accelerator)
                        transformer.train()
                        # TODO: Check if the trainable params are in float32

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break    

    accelerator.end_training()

if __name__ == "__main__":
    main()
