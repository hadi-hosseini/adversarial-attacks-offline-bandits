from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance
from src.ucb import UCBAlgorithm
from src.adversary import FindPerturbation, find_random_perturbation

cfg = COFNIG()
k, d, T = cfg.k, cfg.d, cfg.T

if cfg.creat_new_instance:
    mu, logged_data = create_bandit_instance(k, d, cfg.n_samples, cfg.sigma)
    save_bandit_instance(mu, logged_data, cfg.bandit_instance_path)
else:
    mu, logged_data = load_bandit_instance(cfg.bandit_instance_path)

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



run_ucb_without_perturbation(k, d, T, mu, logged_data)
print(60*'=')
print(60*'=')
run_ucb_with_random_perturbation(k, d, T, mu, logged_data, epsilon_attack=0.5)
print(60*'=')
print(60*'=')
