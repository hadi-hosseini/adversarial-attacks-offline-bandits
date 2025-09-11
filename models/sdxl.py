import json
import os
from diffusers import DiffusionPipeline
import torch


with open("prompts.json", "r") as f:
    prompts = json.load(f)

seeds = list(range(100))
device = "cuda"

for item in prompts:
    prompt_id = item['prompt_id']
    prompt = item['prompt']
    for seed in seeds:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to(device)
                
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt, width=512, height=512, generator=generator).images[0]

        save_dir = f"models/sdxl/{prompt_id}"
        os.makedirs(save_dir, exist_ok=True)
        image.save(f"{save_dir}/{seed}.png")
