import numpy as np
import random
from tqdm import tqdm
import torch
class EpsilonGreedyAlgorithm:
    def __init__(self, k, d, T, true_means, logged_data, perturbation, reward_model=None):
        self.k = k
        self.d = d
        self.T = T
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = 0.1
        self.epsilon_min = 0.01

        self.N = np.zeros(k)
        self.empirical_rewards = np.zeros(k)
        self.perturbation = perturbation
        self.reward_model = reward_model

    def get_reward(self, x):
        if self.reward_model:
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0).to(self.device)   
            with torch.no_grad():
                reward = self.reward_model(x).item()
            return reward
        return np.dot(self.true_means[0] + self.perturbation, x)

    def select_arm(self, t):
        epsilon_t = max(self.epsilon_min, self.epsilon / (t+1))

        if t < self.k:
            return t
        else:
          if random.random() < epsilon_t:
            return random.randint(0, self.k - 1)
          else:
            return np.argmax(self.empirical_rewards)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.empirical_rewards[arm] = self.empirical_rewards[arm] + (reward - self.empirical_rewards[arm])/self.N[arm]


    def run(self):
        chosen_arms = np.zeros(self.T, dtype=int)

        for t in tqdm(range(self.T)):
            arm = self.select_arm(t)
            sample = self.logged_data[arm][int(self.N[arm])]
            reward = self.get_reward(sample)
            self.update(arm, reward)

            chosen_arms[t] = arm

        return chosen_arms