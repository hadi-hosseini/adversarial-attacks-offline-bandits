import numpy as np
import torch

from ..reward_architecture import load_params_to_new_model
from ..adversary import FindPerturbationUCB, OSA, OSAImageReward, OSARandomRewardModel, FullTrajectoryUCBAlgorithmImageReward, TrajectoryFreeUCBAlgorithmImageReward
from ..ucb import UCBAlgorithm, UCBAlgorithmImageReward, UCBAlgorithmRandomRewardModel
from ..utils import print_norms
from ..vis import plot_high_dim_vectors


def shuffle_samples(data, seed=42):
    np.random.seed(seed)
    K, n_samples, d = data.shape
    shuffled = np.empty_like(data)
    
    for k in range(K):
        idx = np.arange(n_samples)
        np.random.shuffle(idx)   # shuffle samples
        shuffled[k] = data[k, idx, :]
    
    return shuffled

# run UCB with created perturbation
def run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=None, real_data=False):
    ucb_with_perturb = UCBAlgorithm(k, d, mu, logged_data, perturbation, reward_model=reward_model, real_data=real_data)
    rewards, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1_ucb(k, d, T, mu, empirical_mus, logged_data, epsilon_attack, reward_model=None, real_data=True):
    find_perturbation = FindPerturbationUCB(k, d, empirical_mus, logged_data, epsilon_attack, qp=False, reward_model=reward_model, real_data=real_data)
    chosen_arms = find_perturbation.run(T, mode=2) # mode=1 all conditions; mode=2 just optimal conditions
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    if reward_model is None:
        current_reward_model = None
    else:
        param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
        current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation, device='cuda'))
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=current_reward_model, real_data=real_data)


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

def heuristic1_ucb(k, d, T, mu, empirical_mu, logged_data, epsilon_attack, qp=False, reward_model=None, use_defense=False, real_data=False):
    find_perturbation = OSA(k, d, T, empirical_mu, logged_data, epsilon_attack, qp=qp, reward_model=reward_model, real_data=real_data)
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

    if use_defense:
        np_logged_data = shuffle_samples(np.array(logged_data))
        logged_data = np_logged_data.tolist()
    
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation, reward_model=current_reward_model, real_data=real_data)

    if reward_model is None:
        plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=2, filename="heuristic1", data=mu[1:])
        plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=3, filename="heuristic1", data=mu[1:])


#### Image Reward ####

def full_trajectory_image_reward(k, d, T, logged_data, epsilon_attack, mlp=None, model=None, backbone=None, prompt=None):
    find_perturbation = FullTrajectoryUCBAlgorithmImageReward(k, d, logged_data, epsilon_attack, qp=False, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
    chosen_arms = find_perturbation.run(T)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    param_flat = torch.cat([p.view(-1) for p in mlp.parameters()])
    current_reward_model = load_params_to_new_model(mlp, param_flat + torch.tensor(perturbation, device='cuda'))
    ucb_with_perturb = UCBAlgorithmImageReward(k, d, logged_data, perturbation, mlp=current_reward_model, model=model, backbone=backbone, prompt=prompt)
    _, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

def trajectory_free_image_reward(k, d, T, logged_data, epsilon_attack, mlp=None, model=None, backbone=None, prompt=None):
    find_perturbation = TrajectoryFreeUCBAlgorithmImageReward(k, d, logged_data, epsilon_attack, qp=False, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
    chosen_arms = find_perturbation.run(T)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)

    param_flat = torch.cat([p.view(-1) for p in mlp.parameters()])
    current_reward_model = load_params_to_new_model(mlp, param_flat + torch.tensor(perturbation, device='cuda'))
    ucb_with_perturb = UCBAlgorithmImageReward(k, d, logged_data, perturbation, mlp=current_reward_model, model=model, backbone=backbone, prompt=prompt)
    _, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

def osa_ucb_image_reward(k, d, T, logged_data, epsilon_attack, qp=False, mlp=None, model=None, backbone=None, prompt=None):
    find_perturbation = OSAImageReward(k, d, T, logged_data, epsilon_attack, qp=qp, mlp=mlp, model=model, backbone=backbone, prompt=prompt)
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)
    
    param_flat = torch.cat([p.view(-1) for p in mlp.parameters()])
    current_reward_model = load_params_to_new_model(mlp, param_flat + torch.tensor(perturbation if perturbation is not None else 0.0, device='cuda'))

    ucb_with_perturb = UCBAlgorithmImageReward(k, d, logged_data, perturbation, mlp=current_reward_model, model=model, backbone=backbone, prompt=prompt)
    _, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)


### Random Reward Model ###

def osa_ucb_random_reward_model(k, d, T, logged_data, epsilon_attack, qp=False, reward_model=None):
    find_perturbation = OSARandomRewardModel(k, d, T, logged_data, epsilon_attack, qp=qp, reward_model=reward_model)
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)
    
    param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
    current_reward_model = load_params_to_new_model(reward_model, param_flat + torch.tensor(perturbation if perturbation is not None else 0.0, device='cuda'))

    ucb_with_perturb = UCBAlgorithmRandomRewardModel(k, d, logged_data=logged_data, perturbation=perturbation, reward_model=current_reward_model)
    _, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)