from ..adversary import FindPerturbationETC
from ..etc import ETCAlgorithm
from ..utils import print_norms

# run ETC with created perturbation
def run_etc_with_created_perturbation(k, d, m, T, mu, logged_data, perturbation):
    etc = ETCAlgorithm(k, m, d, mu, logged_data, perturbation)
    rewards, chosen_arms = etc.run(T)
    print("\nNumber of pulls per arm:", etc.N)
    print(chosen_arms)

# ablation 1 (check all inequalities)
def ablation1_etc(k, d, m, T, mu, logged_data, epsilon_attack, target_arm):
    find_perturbation = FindPerturbationETC(k, m, d, target_arm, mu, logged_data, epsilon_attack, qp=False)
    chosen_arms = find_perturbation.run(T)
    print("\nNumber of pulls per arm:", find_perturbation.N)
    print(chosen_arms)
    perturbation = find_perturbation.perturbation
    print_norms(perturbation)
    run_etc_with_created_perturbation(k, d, m, T, mu, logged_data, perturbation)