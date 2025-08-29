from ..adversary import FindPerturbationETC
from ..etc import ETCAlgorithm
from ..utils import print_norms
import torch
from ..reward_architecture import load_params_to_new_model

# run ETC with created perturbation
def run_etc_with_created_perturbation(k, d, m, T, mu, logged_data, perturbation, reward_model=None):
    etc = ETCAlgorithm(k, m, d, mu, logged_data, perturbation, reward_model)
    rewards, chosen_arms = etc.run(T)
    print("\nNumber of pulls per arm:", etc.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1_etc(k, d, m, T, mu, empirical_mus, logged_data, epsilon_attack, target_arm, reward_model=None):
    find_perturbation = FindPerturbationETC(k, m, d, target_arm, empirical_mus, logged_data, epsilon_attack, qp=False, reward_model=reward_model)
    chosen_arms = find_perturbation.run(T)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    if reward_model is None:
        current_reward_model = None
    else:
        param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
        current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation, device='cuda'))

    run_etc_with_created_perturbation(k, d, m, T, mu, logged_data, perturbation, current_reward_model)