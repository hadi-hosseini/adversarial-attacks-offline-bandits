import math
import cvxpy as cp
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

class UCBAlgorithm:
    def __init__(self, k, d, true_means, logged_data, perturbation):
        self.k = k
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
        if t < self.k:
            return t

        ucb_values = np.zeros(self.k)
        for j in range(self.k):
            mean_term = self.empirical_rewards[j]
            confidence_bound = math.sqrt((2 * math.log(t)) / self.N[j])

            ucb_values[j] = mean_term + confidence_bound

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.total_rewards[arm] += reward
        self.empirical_rewards[arm] = self.total_rewards[arm] / self.N[arm]

    def run(self, T):
        rewards = np.zeros(T)
        chosen_arms = np.zeros(T, dtype=int)

        for t in tqdm(range(T)):
            arm = self.select_arm(t)
            sample = self.logged_data[arm][int(self.N[arm])]
            reward = self.get_reward(sample)

            self.update(arm, reward)
            self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]

            rewards[t] = reward
            chosen_arms[t] = arm

        return rewards, chosen_arms
