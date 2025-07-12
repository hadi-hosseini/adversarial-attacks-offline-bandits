from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance
from src.ucb import UCBAlgorithm
from src.etc import ETCAlgorithm
from src.adversary import find_random_perturbation
from src.ablations.ucb_ablations import *
from src.ablations.etc_ablations import *

cfg = COFNIG()
k, d, T, m, attack_algorithm = cfg.k, cfg.d, cfg.T, cfg.m, cfg.attack_algorithm

print(f"Bandit Instance with  T:{T}, K:{k}, D:{d}")

if cfg.creat_new_instance:
    mu, logged_data = create_bandit_instance(k, d, cfg.n_samples, cfg.sigma)
    save_bandit_instance(mu, k, d, logged_data, cfg.bandit_instance_path)
    print("Save new Bandit Instance")
else:
    mu, logged_data = load_bandit_instance(cfg.bandit_instance_path)
    print("Load Bandit Instance")


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


    run_ucb_without_perturbation(k, d, T, mu, logged_data) # run ucb without perturbation
    print(60*'=')
    print(60*'=')

    # run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack=0.5) # run ucb with random perturbation
    # print(60*'=')
    # print(60*'=')

    # print("ABLATION 1")
    # ablation1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5) # check all inequalities
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

    print("HEURISTIC 1")
    heuristic1_ucb(k, d, T, mu, logged_data, epsilon_attack=0.5, qp=False, hijack_traj=True)
    print(60*'=')
    print(60*'=')


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