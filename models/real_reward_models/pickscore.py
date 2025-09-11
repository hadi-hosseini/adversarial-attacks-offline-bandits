import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs_and_grad(prompt, image_path):
    # Preprocess
    image = Image.open(image_path)
    image_inputs = processor(
        images=[image],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    # Forward pass to get embeddings
    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    # Get text hidden states (before projection)
    text_outputs = model.text_model(**text_inputs)
    pooled_text = text_outputs.last_hidden_state[:, -1, :]  # [batch, hidden_dim]

    # Enable gradient on pooled_text
    pooled_text.requires_grad_(True)

    # Pass through projection
    text_embs = model.text_projection(pooled_text)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # Compute similarity and softmax
    scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    probs = torch.softmax(scores, dim=-1)

    # Select the probability of the first (or target) image
    target_prob = probs[0]

    # Compute gradient w.r.t pooled_text
    grad = torch.autograd.grad(target_prob, pooled_text)[0]

    return target_prob.item(), grad

# Example usage
image_path = "data/generative_models/kandinsky/1/0.png"
prompt = "In the park, a statue stands in the middle, surrounded by blooming flowers."
prob, grad = calc_probs_and_grad(prompt, image_path)

print("Probability:", prob)
print("Gradient shape:", grad.shape)
