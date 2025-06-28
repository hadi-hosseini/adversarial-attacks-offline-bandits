from dataclasses import dataclass

@dataclass
class COFNIG:
    k: int = 3  # The number of arms
    d: int = 1000  # The dimension of attack and mus
    n_samples: int = 3000  # The number of samples for each arm in the logged data
    sigma: float = 1.0  # The variance of Gaussian distribution of each arm
    T: int = 100  # The number of rounds
    creat_new_instance: bool = False # Whether to create a new bandit instance or not
    bandit_instance_path: str = "./data/bandit_data.npz"