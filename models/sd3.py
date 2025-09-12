import torch
from diffusers import StableDiffusion3Pipeline
import huggingface_hub
import os
import json

token = "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"

from huggingface_hub import login
login(token=token)

device = 'cuda'

with open("prompts.json", "r") as f:
    prompts = json.load(f)

seeds = list(range(100))

for item in prompts:
    prompt_id = item['prompt_id']
    prompt = item['prompt']
    for seed in seeds:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)

        generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            width=512,
            height=512,
            generator=generator
        ).images[0]


        save_dir = f"models/sd3/{prompt_id}"
        os.makedirs(save_dir, exist_ok=True)
        image.save(f"{save_dir}/{seed}.png")
