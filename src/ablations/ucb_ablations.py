import numpy as np
import torch

from ..reward_architecture import load_params_to_new_model
from ..adversary import FindPerturbationUCB, OSA
from ..ucb import UCBAlgorithm
from ..utils import print_norms
from ..vis import plot_high_dim_vectors

# run UCB with created perturbation
def run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=None):
    ucb_with_perturb = UCBAlgorithm(k, d, mu, logged_data, perturbation, reward_model=reward_model)
    rewards, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1_ucb(k, d, T, mu, empirical_mus, logged_data, epsilon_attack, reward_model=None):
    find_perturbation = FindPerturbationUCB(k, d, empirical_mus, logged_data, epsilon_attack, qp=False, reward_model=reward_model)
    chosen_arms = find_perturbation.run(T, mode=1) # mode=1 all conditions; mode=2 just optimal conditions
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    if reward_model is None:
        current_reward_model = None
    else:
        param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
        current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation, device='cuda'))
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=current_reward_model)

# ablation 5 (targeted attack)
def ablation5_ucb(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

 # Infinity norm attack

def heuristic1_ucb(k, d, T, mu, empirical_mu, logged_data, epsilon_attack, qp=False, reward_model=None):
    find_perturbation = OSA(k, d, T, empirical_mu, logged_data, epsilon_attack, qp=qp, reward_model=reward_model)
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)
    
    if reward_model is None:
        current_reward_model = None
    else:
        param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
        current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation if perturbation is not None else 0.0, device='cuda'))

    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=current_reward_model)

    if reward_model is None:
        plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=2, filename="heuristic1", data=mu[1:])
        plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=3, filename="heuristic1", data=mu[1:])