import math
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
import clip

from src.reward_architecture import fw0_and_grad
from models.real_reward_models.image_reward import get_score

class UCBAlgorithm:
    def __init__(self, k, d, true_means, logged_data, perturbation, reward_model=None, device=None, real_data=False):
        self.k = k
        self.d = d
        self.true_means = true_means
        self.logged_data = logged_data

        self.N = np.zeros(k)
        self.total_rewards = np.zeros(k)
        self.empirical_rewards = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.reward_model = reward_model
        self.real_data = real_data

        if self.real_data:
            self.encoder_model, self.encoder_preprocess = clip.load("ViT-B/32", device='cuda')


        if isinstance(perturbation, float) and self.reward_model is not None:
           w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
           self.perturbation = np.zeros(w)
        else:
            self.perturbation = perturbation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_reward(self, x, t):
        if self.reward_model:
            if not self.real_data:
                x = torch.from_numpy(x).float()
                x = x.unsqueeze(0).to(self.device)  
            else:
                # x = Image.open(x).convert("RGB")
                # transform = T.Compose([T.ToTensor()])
                # x = transform(x)
                # x = x.view(-1)
                x = self.encoder_preprocess(Image.open(x)).unsqueeze(0).to('cuda')
                with torch.no_grad():
                    x = self.encoder_model.encode_image(x).view(-1)
                x = x.unsqueeze(0).to(self.device, dtype=torch.float32) 
            with torch.no_grad():
                reward = self.reward_model(x).item()
            return reward
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
            reward = self.get_reward(sample, t)

            self.update(arm, reward)
            if not self.real_data:
                self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]

            rewards[t] = reward
            chosen_arms[t] = arm

        return rewards, chosen_arms

class UCBAlgorithmRandomRewardModel:
    def __init__(self, k, d, logged_data, perturbation, reward_model=None):
        self.k = k
        self.d = d
        self.logged_data = logged_data

        self.N = np.zeros(k)
        self.total_rewards = np.zeros(k)
        self.empirical_rewards = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.reward_model = reward_model
        self.encoder_model, self.encoder_preprocess = clip.load("ViT-B/32", device='cuda')


        if isinstance(perturbation, float) and self.reward_model is not None:
           w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
           self.perturbation = np.zeros(w)
        else:
            self.perturbation = perturbation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_reward(self, x, t):
        x = self.encoder_preprocess(Image.open(x)).unsqueeze(0).to('cuda')
        with torch.no_grad():
            x = self.encoder_model.encode_image(x).view(-1)
            x = x.unsqueeze(0).to(self.device, dtype=torch.float32) 
            reward = self.reward_model(x).item()
        return reward

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
            reward = self.get_reward(sample, t)

            self.update(arm, reward)

            rewards[t] = reward
            chosen_arms[t] = arm

        return rewards, chosen_arms

class UCBAlgorithmImageReward:
    def __init__(self, k, d, logged_data, perturbation, mlp=None, model=None, backbone=None, prompt=None):
        self.k = k
        self.d = d
        self.logged_data = logged_data

        self.N = np.zeros(k)
        self.total_rewards = np.zeros(k)
        self.empirical_rewards = np.zeros(k)
        self.reward_model = mlp

        self.backbone = backbone
        self.model = model
        self.prompt = prompt

        if isinstance(perturbation, float):
           w = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
           self.perturbation = np.zeros(w)
        else:
            self.perturbation = perturbation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_reward(self, x, t):
        with torch.no_grad():
            return get_score(self.prompt, x, self.backbone, self.reward_model, self.model)

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
            reward = self.get_reward(sample, t)

            self.update(arm, reward)
            rewards[t] = reward
            chosen_arms[t] = arm

        return rewards, chosen_arms
