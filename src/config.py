from dataclasses import dataclass
from typing import Tuple

@dataclass
class COFNIG:
    k: int = 3 # The number of arms
    d: int = 50  # The dimension of attack and mus
    n_samples: int = 3000  # The number of samples for each arm in the logged data
    sigma: float = 1.0  # The variance of Gaussian distribution of each arm
    T: int = 100  # The number of rounds
    m: int = 5 # The number of pulling each arm in ETC algorithm
    creat_new_instance: bool = True # Whether to create a new bandit instance or not
    train_reward_model: bool = True # Train the reward model or not
    # hidden_sizes: Tuple = (16, 4) # Hidden sizes of the Reward Model
    hidden_sizes: Tuple = (1000,) # Hidden sizes of the Reward Model
    is_mse: bool = True # Train reward model as BCE or MSE
    bandit_instance_path = f"./data/bandit_data_k{k}_d{d}_{'mse' if is_mse else 'bce'}.npz" # Path to save the bandit instance
    attack_algorithm: str = "ucb" # ["ucb", "etc", "epsilon_greedy", "thompson sampling"]
    reward_model_save_path: str = f"./data/reward_model_k{k}_d{d}_{'mse' if is_mse else 'bce'}.pt"