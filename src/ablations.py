
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