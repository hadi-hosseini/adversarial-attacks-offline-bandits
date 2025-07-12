from dataclasses import dataclass

@dataclass
class COFNIG:
    k: int = 3 # The number of arms
    d: int = 1000  # The dimension of attack and mus
    n_samples: int = 3000  # The number of samples for each arm in the logged data
    sigma: float = 1.0  # The variance of Gaussian distribution of each arm
    T: int = 100  # The number of rounds
    m: int = 5 # The number of pulling each arm in ETC algorithm
    creat_new_instance: bool = False # Whether to create a new bandit instance or not
    bandit_instance_path = f"./data/bandit_data_k{k}_d{d}.npz" # Path to save the bandit instance
    attack_algorithm: str = "ucb" # ["ucb", "etc", "epsilon_greedy", "thompson sampling"]