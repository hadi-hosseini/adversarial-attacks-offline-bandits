from .ucb import UCBAlgorithm
import numpy as np


def print_norms(x):
    norm_two = np.linalg.norm(x)
    norm_infinity = np.linalg.norm(x, ord=np.inf)
    print("norm 2:", norm_two)
    print("norm infinity: ", norm_infinity)

def print_all_perturbs(k, d, mu, logged_data, all_perturbs):
    for perturb in all_perturbs:
      ucb_with_perturb = UCBAlgorithm(k, d, mu, logged_data, perturb)
      rewards, chosen_arms = ucb_with_perturb.run(T)
      print("\nNumber of pulls per arm:", ucb_with_perturb.N)
      print(chosen_arms)