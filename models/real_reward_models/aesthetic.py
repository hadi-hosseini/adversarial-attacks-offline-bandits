import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import torch
from PIL import Image
import open_clip

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.reward_architecture import fw0_and_grad

def get_aesthetic_mlp(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def get_aesthetic_backbone():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    return model, preprocess

def get_aesthetic_score(model, preprocess, amodel, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to('cuda')

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = amodel(image_features)
    
    return score