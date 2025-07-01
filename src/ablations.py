import numpy as np

from .adversary import FindPerturbation
from .ucb import UCBAlgorithm
from .utils import print_norms

# run UCB with created perturbation
def run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation):
    ucb_with_perturb = UCBAlgorithm(k, d, mu, logged_data, perturbation)
    rewards, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=False)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 2 (check only the optimal arm inequalities to be satisfied)
def ablation2(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=False)
    chosen_arms = find_perturbation.run(T, mode=2)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 3 (Use Quadratic Problem)
def ablation3(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=True)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 4 (Satisfy inequalities only M alternatives)
def ablation4(k, d, T, mu, logged_data, epsilon_attack, M):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=False, M=M)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 5 (targeted attack)
def ablation5(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

 # Infinity norm attack
def ablation6(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    find_perturbation = FindPerturbation(k, d, mu, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm, infinity_attack=True)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)
        
# Restricted Threat Model (Infer Mu)
def ablation7(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    empirical_mus = [np.mean(arm_samples, axis=0) for arm_samples in logged_data]
    error = np.linalg.norm(empirical_mus[0] - mu[0])
    print("L2 distance:", error)
     
    find_perturbation = FindPerturbation(k, d, empirical_mus, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)