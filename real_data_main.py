import torch
import random
import os
import glob
from PIL import Image

from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance
from src.ucb import UCBAlgorithm
from src.etc import ETCAlgorithm
from src.epsilon_greedy import EpsilonGreedyAlgorithm
from src.adversary import find_random_perturbation
from src.reward_architecture import RewardModel
from src.ablations.ucb_ablations import *
from src.ablations.etc_ablations import *
from src.ablations.epsilon_greedy_ablations import *

cfg = COFNIG()
k, d, T, m, attack_algorithm, hidden_sizes, is_mse = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm, cfg.hidden_sizes, cfg.is_mse

print(f"Bandit Instance with  T:{T}, K:{k}, Hidden Size:{hidden_sizes}")
print(60*'=')
print(60*'=')

prompt_id = 5
models = ['sdxl', 'sd1_4', 'openjourney']
logged_data = []

for model in models:
    model_data_path = f"models/{model}/{prompt_id}"
    image_paths = glob.glob(os.path.join(model_data_path, "*.png"))
    image_paths.sort()
    logged_data.append(image_paths)


d = 512
reward_model = RewardModel(d=d, hidden_sizes=hidden_sizes, is_mse=is_mse).to('cuda')

if attack_algorithm == "ucb":
    # run UCB with random perturbation
    # def run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack):
    #     print("run UCB with random perturbation")
    #     random_perturbation = find_random_perturbation(d, epsilon_attack)
    #     ucb = UCBAlgorithm(k, d, mu, logged_data, random_perturbation)
    #     _, chosen_arms = ucb.run(T)
    #     ASR = ((ucb.N[1] + ucb.N[2])/99) * 100
    #     print(ASR)
    #     print("\nNumber of pulls per arm:", ucb.N)
    #     print(chosen_arms)

    
    # run UCB without perturbation and with reward model
    def run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data, perturbation=0.0):
        print("run UCB without perturbation; with reward model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ucb = UCBAlgorithm(k, d, true_means=None, logged_data=logged_data, perturbation=perturbation, reward_model=reward_model, device=device, real_data=True)
        _, chosen_arms = ucb.run(T)
        print("\nNumber of pulls per arm:", ucb.N)
        print(chosen_arms)


    run_ucb_without_perturbation_with_reward_model(k, d, T, logged_data) # run ucb without perturbation; with reward model
    print(60*'=')
    print(60*'=')

    # epsilon_attacks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # for epsilon_attack in epsilon_attacks:
    #     run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack=epsilon_attack) # run ucb with random perturbation
    #     print(60*'=')
    #     print(60*'=')


    # print("Full Trajectory - Attacking the Reward Model")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # ablation1_ucb(k, d, T, mu=None, empirical_mus=None, logged_data=logged_data, epsilon_attack=0.5, reward_model=reward_model, real_data=True)
    # print(60*'=')
    # print(60*'=')


    print("OSA - Attacking the Reward Model")
    # empirical_mus = [np.mean(arm_samples, axis=0) for arm_samples in logged_data]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = cfg.reward_model_save_path
    heuristic1_ucb(k, d, T, mu=None, empirical_mu=None, logged_data=logged_data, epsilon_attack=0.5, qp=False, reward_model=reward_model, real_data=True)
    print(60*'=')
    print(60*'=')