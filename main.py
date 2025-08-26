import torch

from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance
from src.ucb import UCBAlgorithm
from src.etc import ETCAlgorithm
from src.adversary import find_random_perturbation
from src.reward_architecture import load_model, train_reward_model
from src.ablations.ucb_ablations import *
from src.ablations.etc_ablations import *

cfg = COFNIG()
k, d, T, m, attack_algorithm, hidden_sizes, is_mse = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm, cfg.hidden_sizes, cfg.is_mse

print(f"Bandit Instance with  T:{T}, K:{k}, D:{d}")
print(60*'=')
print(60*'=')

if cfg.creat_new_instance:
    mu, logged_data = create_bandit_instance(k, d, cfg.n_samples, cfg.sigma)
    save_bandit_instance(mu, k, d, logged_data, cfg.bandit_instance_path)
    print("Save new Bandit Instance")
else:
    mu, logged_data = load_bandit_instance(cfg.bandit_instance_path)
    print("Load Bandit Instance")


if cfg.train_reward_model:
    train_reward_model(cfg)
    

if attack_algorithm == "ucb":
    # run UCB without perturbation
    def run_ucb_without_perturbation(k, d, T, mu, logged_data, perturbation=0.0):
        print("run UCB without perturbation")
        ucb = UCBAlgorithm(k, d, mu, logged_data, perturbation)
        rewards, chosen_arms = ucb.run(T)
        print("\nNumber of pulls per arm:", ucb.N)
        print(chosen_arms)


    # run UCB with random perturbation
    def run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack):
        print("run UCB with random perturbation")
        random_perturbation = find_random_perturbation(d, epsilon_attack)
        ucb = UCBAlgorithm(k, d, mu, logged_data, random_perturbation)
        rewards, chosen_arms = ucb.run(T)
        print("\nNumber of pulls per arm:", ucb.N)
        print(chosen_arms)

    
    # run UCB without perturbation and with reward model
    def run_ucb_without_perturbation_with_reward_model(k, d, T, mu, logged_data, perturbation=0.0):
        print("run UCB without perturbation; with reward model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        save_path = cfg.reward_model_save_path
        reward_model = load_model(d, save_path, hidden_sizes=hidden_sizes, is_mse=is_mse, device=device)
        ucb = UCBAlgorithm(k, d, mu, logged_data, perturbation, reward_model=reward_model, original_reward_model=reward_model, device=device)
        rewards, chosen_arms = ucb.run(T)
        print("\nNumber of pulls per arm:", ucb.N)
        print(chosen_arms)

    # run_ucb_without_perturbation(k, d, T, mu, logged_data) # run ucb without perturbation
    # print(60*'=')
    # print(60*'=')


    run_ucb_without_perturbation_with_reward_model(k, d, T, mu, logged_data) # run ucb without perturbation; with reward model
    print(60*'=')
    print(60*'=')

    # run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack=0.5) # run ucb with random perturbation
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 1")
    # ablation1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5) # check all inequalities
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 1 - Attacking the Reward Model")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # save_path = cfg.reward_model_save_path
    # reward_model = load_model(d, save_path, hidden_sizes=(512, 256), device=device)
    # W, _ = collapse_weights(reward_model)
    # W = W.detach().cpu().numpy()
    # ablation1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, W=W, reward_model=reward_model) # check all inequalities
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 2")
    # ablation2_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5) # check only the optimal arm's inequalities to be satisfied
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 3")
    # ablation3_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5) # check quadratic problem
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 4")
    # ablation4_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, M=10) # M alternatives attack
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 5")
    # ablation5_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, targeted=True, target_arm=2) # Target Attack
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 6")
    # ablation6_ucb(k, d, T, mu, logged_data, epsilon_attack=1/125, targeted=True, target_arm=2) # Infinity norm attack
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 7")
    # ablation7_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, targeted=True, target_arm=2) # Restricted Threat Model (Infer Mu)
    # print(60*'=')
    # print(60*'=')

    # print("HEURISTIC 1")
    # heuristic1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, qp=False)
    # print(60*'=')
    # print(60*'=')


    print("HEURISTIC 1 - Attacking the Reward Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = cfg.reward_model_save_path
    reward_model = load_model(d, save_path, hidden_sizes=hidden_sizes, is_mse=is_mse, device=device)
    heuristic1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, qp=False, reward_model=reward_model)
    print(60*'=')
    print(60*'=')

    # print("HEURISTIC 2")
    # heuristic2_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, qp=False)
    # print(60*'=')
    # print(60*'=')


    # print("HEURISTIC 2 - Attacking the Reward Model")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # save_path = cfg.reward_model_save_path
    # reward_model = load_model(d, save_path, hidden_sizes=(512, 256), device=device)
    # W, _ = collapse_weights(reward_model)
    # W = W.detach().cpu().numpy()
    # heuristic2_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, qp=False, W=W, reward_model=reward_model)
    # print(60*'=')
    # print(60*'=')


elif attack_algorithm == "etc":
    def run_etc_without_perturbation(k, m, d, T, mu, logged_data, perturbation=0.0):
        print("run ETC without perturbation")
        etc = ETCAlgorithm(k, m, d, mu, logged_data, perturbation)
        rewards, chosen_arms = etc.run(T)
        print("\nNumber of pulls per arm:", etc.N)
        print(chosen_arms)


    run_etc_without_perturbation(k, m, d, T, mu, logged_data) # run etc without perturbation
    print(60*'=')
    print(60*'=')

    print("ABLATION 1")
    ablation1_etc(k, d, m, T, mu, logged_data, epsilon_attack=0.5, target_arm=2) # check all inequalities