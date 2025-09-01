import random

from ..adversary import OSAEpsilonGreedy
from ..epsilon_greedy import EpsilonGreedyAlgorithm
from ..utils import print_norms
import torch
from ..reward_architecture import load_params_to_new_model

# run ETC with created perturbation
def run_epsilon_greedy_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=None):
    random.seed(42)
    epsilon_greedy = EpsilonGreedyAlgorithm(k, d, T, mu, logged_data, perturbation)
    chosen_arms = epsilon_greedy.run()
    print("\nNumber of pulls per arm:", epsilon_greedy.N)
    print(chosen_arms)

# ablation 1 (OSA Epsilon Greedy Version)
def ablation1_epsilon_greedy(k, d, T, mu, empirical_mus, logged_data, epsilon_attack, reward_model=None):
    random.seed(42)
    find_perturbation = OSAEpsilonGreedy(k, d, T, empirical_mus, logged_data, epsilon_attack, qp=False, reward_model=reward_model)
    chosen_arms = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    if reward_model is None:
        current_reward_model = None
    else:
        param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
        current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation, device='cuda'))

    run_epsilon_greedy_with_created_perturbation(k, d, T, mu, logged_data, perturbation, current_reward_model)