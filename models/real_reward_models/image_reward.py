import torch
import ImageReward as reward
from PIL import Image
import os


def encoder(prompt, image, blip_model, model):
        if (type(image).__name__=='list'):
            _, rewards = model.inference_rank(prompt, image)
            return rewards
            
        # text encode
        text_input = blip_model.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(model.device)
        
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            
        image = model.preprocess(pil_image).unsqueeze(0).to(model.device)
        image_embeds = blip_model.visual_encoder(image)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(model.device)
        text_output = blip_model.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,)
        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim) 
        return txt_features   

def get_score(prompt, image, blip_model, mlp_model, model):
        if (type(image).__name__=='list'):
            _, rewards = model.inference_rank(prompt, image)
            return rewards
            
        # text encode
        text_input = blip_model.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(model.device)
        
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            
        image = model.preprocess(pil_image).unsqueeze(0).to(model.device)
        image_embeds = blip_model.visual_encoder(image)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(model.device)
        text_output = blip_model.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,)
        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        rewards = mlp_model(txt_features)
        rewards = (rewards - model.mean) / model.std
        
        return rewards.detach().cpu().numpy().item()

# if __name__ == "__main__":
#     prompt = "In the park, a statue stands in the middle, surrounded by blooming flowers and people enjoying a sunny day."
#     img_path = "/home/hadi/BANDIT/adversarial-bandit/models/image_generative_models/sdxl/1/0.png" 

#     model = reward.load("ImageReward-v1.0")

#     backbone = model.blip
#     mlp = model.mlp

#     # --- Freeze backbone ---
#     for param in backbone.parameters():
#         param.requires_grad = False

#     # --- Make MLP trainable ---
#     for param in mlp.parameters():
#         param.requires_grad = True

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     backbone.to(device)
#     mlp.to(device)
#     model.device = device

#     with torch.no_grad():
#         score = get_score(prompt, img_path, backbone, mlp, model)
#         print(f"Score for '{img_path}': {score:.2f}")
