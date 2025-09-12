import torch
import os
import glob

from src.config import COFNIG
from src.ucb import UCBAlgorithmImageReward
from src.ablations.ucb_ablations import *
import ImageReward as reward
from src.utils import read_json

cfg = COFNIG()
k, d, T, m, attack_algorithm, hidden_sizes, is_mse = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm, cfg.hidden_sizes, cfg.is_mse

print(f"Bandit Instance with  T:{T}, K:{k}, Hidden Size:{hidden_sizes}")
print(60*'=')
print(60*'=')


prompt_id = 1
prompts = read_json("models/prompts.json")
prompt = prompts[prompt_id - 1]['prompt']


models = ['sd1_4', 'openjourney', 'kandinsky', 'sdxl']
logged_data = []

for model in models:
    model_data_path = f"data/generative_models/{model}/{prompt_id}"
    image_paths = glob.glob(os.path.join(model_data_path, "*.png"))
    image_paths.sort()
    logged_data.append(image_paths)


model = reward.load("ImageReward-v1.0")
backbone = model.blip
mlp = model.mlp

for param in backbone.parameters():
    param.requires_grad = False

for param in mlp.parameters():
    param.requires_grad = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    mlp.to(device)
    model.device = device


# run UCB without perturbation and with reward model
def run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data, perturbation=0.0):
    print("run UCB without perturbation; with reward model")
    ucb = UCBAlgorithmImageReward(k, d, logged_data=logged_data, perturbation=perturbation, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
    _, chosen_arms = ucb.run(T)
    print("\nNumber of pulls per arm:", ucb.N)
    print(chosen_arms)


print("Original UCB Method")
run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data)
print(60*'=')
print(60*'=')


# print("Full Trajectory - Attacking")
# full_trajectory_image_reward(k, d, T, logged_data=logged_data, epsilon_attack=0.5, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
# print(60*'=')
# print(60*'=')

# print("Trajectory-Free - Attacking")
# trajectory_free_image_reward(k, d, T, logged_data=logged_data, epsilon_attack=0.5, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
# print(60*'=')
# print(60*'=')


print("OSA - Attacking")
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = cfg.reward_model_save_path
osa_ucb_image_reward(k, d, T, logged_data=logged_data, epsilon_attack=10.0, qp=False, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
print(60*'=')
print(60*'=')