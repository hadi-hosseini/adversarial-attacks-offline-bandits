import numpy as np

from ..adversary import FindPerturbationUCB, AdaptivePerturbationUCB
from ..ucb import UCBAlgorithm
from ..utils import print_norms
from ..vis import plot_high_dim_vectors

# run UCB with created perturbation
def run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation):
    ucb_with_perturb = UCBAlgorithm(k, d, mu, logged_data, perturbation)
    rewards, chosen_arms = ucb_with_perturb.run(T)
    print("\nNumber of pulls per arm:", ucb_with_perturb.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1_ucb(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=False)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 2 (check only the optimal arm inequalities to be satisfied)
def ablation2_ucb(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=False)
    chosen_arms = find_perturbation.run(T, mode=2)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 3 (Use Quadratic Problem)
def ablation3_ucb(k, d, T, mu, logged_data, epsilon_attack):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=True)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

# ablation 4 (Satisfy inequalities only M alternatives)
def ablation4_ucb(k, d, T, mu, logged_data, epsilon_attack, M):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=False, M=M)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

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
def ablation6_ucb(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    find_perturbation = FindPerturbationUCB(k, d, mu, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm, infinity_attack=True)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)
        
# Restricted Threat Model (Infer Mu)
def ablation7_ucb(k, d, T, mu, logged_data, epsilon_attack, targeted, target_arm):
    empirical_mus = [np.mean(arm_samples, axis=0) for arm_samples in logged_data]
    error = np.linalg.norm(empirical_mus[0] - mu[0])
    print("L2 distance:", error)
     
    find_perturbation = FindPerturbationUCB(k, d, empirical_mus, logged_data, epsilon_attack, qp=False, targeted=targeted, target_arm=target_arm)
    chosen_arms = find_perturbation.run(T, mode=1)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

def heuristic1_ucb(k, d, T, mu, logged_data, epsilon_attack, qp=False):
    find_perturbation = AdaptivePerturbationUCB(k, d, T, mu, logged_data, epsilon_attack, qp=qp)
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)

    plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=2, filename="heuristic1", data=mu[1:])
    plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=3, filename="heuristic1", data=mu[1:])

def heuristic2_ucb(k, d, T, mu, logged_data, epsilon_attack, qp=False):
    empirical_mus = [np.mean(arm_samples, axis=0) for arm_samples in logged_data]
    dot_products = [np.dot(emp_mu, mu[0]) for emp_mu in empirical_mus]
    sorted_indices = np.argsort(dot_products)
    runner_up_arm = sorted_indices[-2]

    find_perturbation = AdaptivePerturbationUCB(k, d, T, mu, logged_data, epsilon_attack, qp=qp, target_arm=runner_up_arm)
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)
    run_ucb_with_created_perturbation(k, d, T, mu, logged_data, perturbation)


    plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=2, filename="heuristic2", data=mu[1:])
    plot_high_dim_vectors(mu[0], perturbation + mu[0], dim=3, filename="heuristic2", data=mu[1:])

