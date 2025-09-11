import json
import torch
from diffusers import StableDiffusionPipeline
import os


with open("models/prompts.json", "r") as f:
    prompts = json.load(f)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

seeds = list(range(100))

for item in prompts[10:]:
    prompt_id = item['prompt_id']
    prompt = item['prompt']
    for seed in seeds:
        out_dir = f"models/sd1_4/{prompt_id}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{seed}.png"

        if os.path.exists(out_path):
            continue

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)

        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt, generator=generator).images[0]

        image.save(out_path)
        print(f"Saved {out_path}")