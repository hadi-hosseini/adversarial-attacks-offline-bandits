from dataclasses import dataclass
from typing import Tuple

@dataclass
class COFNIG:
    k: int = 4 # The number of arms # k = [3, 5, 10]
    d: int = 1000  # The dimension of attack and mus # d = [10000, 1000, 100, 10] 
    n_samples: int = 3000  # The number of samples for each arm in the logged data 
    sigma: float = 1.0  # The variance of Gaussian distribution of each arm
    T: int = 100  # The number of rounds # T = [100, 50]
    m: int = 5 # The number of pulling each arm in ETC algorithm
    creat_new_instance: bool = False # Whether to create a new bandit instance or not
    train_reward_model: bool = False # Train the reward model or not
    # hidden_sizes: Tuple = (16, 4) # Hidden sizes of the Reward Model
    hidden_sizes: Tuple = (4000,) # Hidden sizes of the Reward Model # hidden_size = [50, 100, 1000, 2000, 5000, 8000, 10000]
    is_mse: bool = True # Train reward model as BCE or MSE
    bandit_instance_path = f"./data/bandit_data_k{k}_d{d}_{'mse' if is_mse else 'bce'}_T{T}.npz" # Path to save the bandit instance
    attack_algorithm: str = "ucb" # ["ucb", "etc", "epsilon_greedy", "thompson sampling"]
    reward_model_save_path: str = f"./data/reward_model_k{k}_d{d}_{'mse' if is_mse else 'bce'}_T{T}.pt"