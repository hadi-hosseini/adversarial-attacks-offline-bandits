import torch
import os
import glob

from src.config import COFNIG
from src.ucb import UCBAlgorithmAesthetic
from src.ablations.ucb_ablations import *
from models.real_reward_models.aesthetic import get_aesthetic_mlp, get_aesthetic_backbone
from src.utils import save_json

cfg = COFNIG()
k, d, T, m, attack_algorithm, hidden_sizes, is_mse = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm, cfg.hidden_sizes, cfg.is_mse

print(f"Bandit Instance with  T:{T}, K:{k}, Hidden Size:{hidden_sizes}")
print(60*'=')
print(60*'=')


models = ['openjourney', 'sd1_4', 'kandinsky', 'sdxl']

def create_logged_data(prompt_id):
    logged_data = []
    for model_name in models:
        model_data_path = f"data/generative_models/{model_name}/{prompt_id}"
        image_paths = glob.glob(os.path.join(model_data_path, "*.png"))
        image_paths.sort()
        logged_data.append(image_paths)
    return logged_data


mlp = get_aesthetic_mlp(clip_model="vit_l_14")
model, preprocess = get_aesthetic_backbone()

for param in model.parameters():
    param.requires_grad = False

for param in mlp.parameters():
    param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
mlp.to(device)

# run UCB without perturbation and with reward model
def run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data, perturbation=0.0):
    print("run UCB without perturbation; with reward model")
    ucb = UCBAlgorithmAesthetic(k, d, logged_data=logged_data, perturbation=perturbation, mlp=mlp, model=model, preprocess=preprocess)
    _, chosen_arms = ucb.run(T)
    print("\nNumber of pulls per arm:", ucb.N)
    best_arm = np.argmax(ucb.N)
    print(chosen_arms)
    return best_arm


asr_results = dict()
for prompt_id in range(1, 16, 1):
    logged_data = create_logged_data(prompt_id)
    print("Original UCB Method")
    best_arm = run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data)
    print(60*'=')
    print(60*'=')

    print("OSA - Attacking")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = cfg.reward_model_save_path
    ASR = osa_ucb_aesthetic(k, d, T, logged_data=logged_data, epsilon_attack=6.0, qp=False, mlp=mlp, model=model, preprocess=preprocess, best_arm=best_arm) # 10/28.66 ~ 0.34 norm-2 attack
    print(60*'=')
    print(60*'=')

    asr_results[prompt_id] = ASR
    save_json(asr_results, f"results/aesthetic_model_results_T{T}_K{k}.json")


# print("Full Trajectory - Attacking")
# full_trajectory_image_reward(k, d, T, logged_data=logged_data, epsilon_attack=0.5, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
# print(60*'=')
# print(60*'=')

# print("Trajectory-Free - Attacking")
# trajectory_free_image_reward(k, d, T, logged_data=logged_data, epsilon_attack=0.5, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
# print(60*'=')
# print(60*'=')