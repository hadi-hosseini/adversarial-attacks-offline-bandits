from diffusers import StableDiffusionPipeline
import torch
import json
import os

with open("models/prompts.json", "r") as f:
    prompts = json.load(f)

seeds = list(range(100))
device = "cuda"


for item in prompts[20:]:
    prompt_id = item['prompt_id']
    prompt = item['prompt']
    for seed in seeds:
        model_id = "prompthero/openjourney"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt, generator=generator).images[0]

        save_dir = f"data/generative_models/openjourney/{prompt_id}"
        os.makedirs(save_dir, exist_ok=True)
        image.save(f"{save_dir}/{seed}.png")
