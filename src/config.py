from dataclasses import dataclass
from typing import Tuple

@dataclass
class COFNIG:
    k: int = 4 # The number of arms # K=3
    d: int = 1000  # The dimension of attack and mus  # D=1000
    n_samples: int = 3000  # The number of samples for each arm in the logged data 
    sigma: float = 1.0  # The variance of Gaussian distribution of each arm
    T: int = 50  # The number of rounds # T=100000
    m: int = 5 # The number of pulling each arm in ETC algorithm
    creat_new_instance: bool = False # Whether to create a new bandit instance or not
    train_reward_model: bool = False # Train the reward model or not
    # hidden_sizes: Tuple = (16, 4) # Hidden sizes of the Reward Model
    hidden_sizes: Tuple = (1000,) # Hidden sizes of the Reward Model 
    is_mse: bool = True # Train reward model as BCE or MSE
    bandit_instance_path = f"./data/bandit_data_d{d}_T{T}.npz" # Path to save the bandit instance
    attack_algorithm: str = "ucb" # ["ucb", "etc", "epsilon_greedy", "thompson sampling"]
    reward_model_save_path: str = f"./data/reward_model_d{d}_T{T}.pt"