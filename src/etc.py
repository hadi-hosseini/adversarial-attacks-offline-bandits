import numpy as np
from tqdm import tqdm

class ETCAlgorithm:
    def __init__(self, k, m, d, true_means, logged_data, perturbation):
        self.k = k
        self.m = m
        self.d = d
        self.true_means = true_means
        self.logged_data = logged_data

        self.N = np.zeros(k)
        self.total_rewards = np.zeros(k)
        self.empirical_rewards = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.perturbation = perturbation

    def get_reward(self, x):
        return np.dot(self.true_means[0] + self.perturbation, x)

    def select_arm(self, t):
        if t < self.k * self.m:
            return t % self.k

        return np.argmax(self.empirical_rewards)

    def update(self, arm, reward):
        self.total_rewards[arm] += reward
        self.empirical_rewards[arm] = self.total_rewards[arm] / self.N[arm]

    def run(self, T):
        rewards = np.zeros(T)
        chosen_arms = np.zeros(T, dtype=int)

        for t in tqdm(range(T)):
            arm = self.select_arm(t)
            sample = self.logged_data[arm][int(self.N[arm])]
            reward = self.get_reward(sample)
            self.N[arm] += 1

            if t < self.k * self.m:
              self.update(arm, reward)
              self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]

            rewards[t] = reward
            chosen_arms[t] = arm

        return rewards, chosen_arms