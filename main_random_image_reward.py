import torch
import os
import glob

from src.config import COFNIG
from src.ucb import UCBAlgorithmRandomRewardModel
from src.reward_architecture import RewardModel
from src.ablations.ucb_ablations import *
from src.utils import read_json, set_seed

cfg = COFNIG()
k, d, T, m, attack_algorithm, hidden_sizes, is_mse = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm, cfg.hidden_sizes, cfg.is_mse

print(f"Bandit Instance with  T:{T}, K:{k}, Hidden Size:{hidden_sizes}")
print(60*'=')
print(60*'=')

prompt_id = 3
prompts = read_json("models/prompts.json")
prompt = prompts[prompt_id - 1]['prompt']

models = ['openjourney', 'sd1_4', 'kandinsky', 'sdxl']
logged_data = []

for model in models:
    model_data_path = f"data/generative_models/{model}/{prompt_id}"
    image_paths = glob.glob(os.path.join(model_data_path, "*.png"))
    image_paths.sort()
    logged_data.append(image_paths)


d = 512
set_seed()
reward_model = RewardModel(d=d, hidden_sizes=hidden_sizes, is_mse=is_mse).to('cuda')

if attack_algorithm == "ucb":
    
    # run UCB without perturbation and with reward model
    def run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data, perturbation=0.0):
        print("run UCB without perturbation; with reward model")
        ucb = UCBAlgorithmRandomRewardModel(k, d, logged_data=logged_data, perturbation=perturbation, reward_model=reward_model)
        _, chosen_arms = ucb.run(T)
        print("\nNumber of pulls per arm:", ucb.N)
        print(chosen_arms)


    run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data) # run ucb without perturbation; with reward model
    print(60*'=')
    print(60*'=')


    # print("Full Trajectory Attacking")
    # full_trajectory_random_reward_model(k, d, T, logged_data=logged_data, epsilon_attack=0.5, reward_model=reward_model)
    # print(60*'=')
    # print(60*'=')

    # print("Trajectory-Free Attacking")
    # trajectory_free_random_reward_model(k, d, T, logged_data=logged_data, epsilon_attack=0.5, reward_model=reward_model)
    # print(60*'=')
    # print(60*'=')


    print("OSA Attacking")
    save_path = cfg.reward_model_save_path
    osa_random_reward_model(k, d, T, logged_data=logged_data, epsilon_attack=0.5, qp=False, reward_model=reward_model)
    print(60*'=')
    print(60*'=')