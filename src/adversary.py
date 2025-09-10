import numpy as np
from tqdm import tqdm
import math
import cvxpy as cp
import torch
import copy
import random
import torchvision.transforms as Transform
from PIL import Image
import clip

from src.reward_architecture import fw0_and_grad, load_params_to_new_model


def find_random_perturbation(d, epsilon):
    np.random.seed(100)
    perturbation = np.random.randn(d)
    perturbation = epsilon * perturbation / np.linalg.norm(perturbation)
    return perturbation


class FindPerturbationUCB:
    def __init__(self, k, d, true_means, logged_data, epsilon, qp=False, targeted=False, target_arm=1, reward_model=None, real_data=False):
        self.k = k
        self.d = d
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = epsilon
        self.qp = qp
        self.targeted = targeted
        self.reward_model = reward_model
        self.real_data = real_data
        if self.real_data: 
          self.encoder_model, self.encoder_preprocess = clip.load("ViT-B/32", device='cuda')

        if self.reward_model is not None:
          self.param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
          w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
          print(f"Number of all params of the network is: {w}")
          d = w
          self.d = w
          self.empirical_f = np.zeros(k)
          self.empirical_grad = np.zeros((k, d))

        self.N = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.perturbation = None
        self.history = []
        self.all_perturbs = []
        self.target_arm = target_arm
        self.turn = 1

    def select_arm(self, t):
        if t < self.k:
            return t
        
        ### targetted
        if self.targeted:
          return self.target_arm

        ### untargetted
        else:
          self.turn += 1
          if self.turn == self.k:
            self.turn = 1
          return self.turn

    def find_perturbation_optimal_only(self, arm, t):
        x = cp.Variable(self.d)

        if self.reward_model is None:
          d_0 = self.empirical_means[arm] - self.empirical_means[0]
          c_0 = (math.sqrt(2 * math.log(t) / self.N[0]) - math.sqrt(2 * math.log(t) / self.N[arm])) - np.dot(self.true_means[0], d_0)
        else:
          d_0 = self.empirical_grad[arm] - self.empirical_grad[0]
          c_0 = (math.sqrt((2 * math.log(t)) / self.N[0]) - math.sqrt((2 * math.log(t)) / self.N[arm])) + (self.empirical_f[0] - self.empirical_f[arm])
        self.history.append((d_0, c_0))

        constraints = []
        for (d_0, c_0) in self.history:
            constraints.append(x @ d_0 >= c_0 + 1e-6)
        constraints.append(cp.norm(x, 2) <= self.epsilon)
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()

        if prob.status == 'optimal':
          self.all_perturbs.append(x.value)
          return x.value
        else:
          print("don't fine perturbation")
          return None

    def find_perturbation(self, arm, t):
        x = cp.Variable(self.d)

        for j in range(self.k):
            if j != arm:
              if self.reward_model is None:
                d_j = self.empirical_means[arm] - self.empirical_means[j]
                c_j = (math.sqrt((2 * math.log(t)) / self.N[j]) - math.sqrt((2 * math.log(t)) / self.N[arm])) - np.dot(self.true_means[0], d_j)
              else:
                d_j = self.empirical_grad[arm] - self.empirical_grad[j]
                c_j = (math.sqrt((2 * math.log(t)) / self.N[j]) - math.sqrt((2 * math.log(t)) / self.N[arm])) + (self.empirical_f[j] - self.empirical_f[arm])
              self.history.append((d_j, c_j))

        constraints = []
        for (d_j, c_j) in self.history:
            constraints.append(x @ d_j >= c_j + 1e-6)
        if self.qp:
          objective = cp.Minimize(cp.norm(x, 2))
          prob = cp.Problem(objective, constraints)
        else:
          constraints.append(cp.norm(x, 2) <= self.epsilon)
          prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()

        if prob.status == 'optimal':
          self.all_perturbs.append(x.value)
          return x.value
        else:
          return None


    def run(self, T, mode=1):
        chosen_arms = np.zeros(T, dtype=int)

        for t in tqdm(range(T)):
            arm = self.select_arm(t)

            if t >= self.k:
              if mode == 1: # check all inequalities
                perturbation = self.find_perturbation(arm, t)
              elif mode == 2: # check just optimal inequalities
                perturbation = self.find_perturbation_optimal_only(arm, t)


              if perturbation is None:
                return chosen_arms

              self.perturbation = perturbation

            sample = self.logged_data[arm][int(self.N[arm])]


            if self.real_data:
              image = self.encoder_preprocess(Image.open(sample)).unsqueeze(0).to('cuda')
              with torch.no_grad():
                sample = self.encoder_model.encode_image(image).view(-1)

            self.N[arm] += 1

            if self.reward_model is not None:
              f_x, grad_x = fw0_and_grad(self.reward_model, torch.tensor(sample, dtype=torch.float32, device='cuda'))
              grad_x = grad_x.detach().cpu().numpy()
              self.empirical_f[arm] = self.empirical_f[arm] + (f_x - self.empirical_f[arm])/self.N[arm]
              self.empirical_grad[arm] = self.empirical_grad[arm] + (grad_x - self.empirical_grad[arm])/self.N[arm]
            else:
              self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]

            chosen_arms[t] = arm

        return chosen_arms
    

class OSA:
    def __init__(self, k, d, T, true_means, logged_data, epsilon, qp=False, target_arm=None, reward_model=None, real_data=False):
        self.k = k
        self.d = d
        self.T = T
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = epsilon
        self.qp = qp
        self.target_arm = target_arm
        self.reward_model = reward_model
        self.real_data = real_data
        if self.real_data: 
          self.encoder_model, self.encoder_preprocess = clip.load("ViT-B/32", device='cuda')
        if self.reward_model is not None:
          self.param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
          w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
          print(f"Number of all params of the network is: {w}")
          d = w
          self.d = w
          self.empirical_f = np.zeros(k)
          self.empirical_grad = np.zeros((k, d))
          self.current_reward_model = copy.deepcopy(self.reward_model)

        self.data = [[] for _ in range(k)]
        self.N = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.empirical_rewards = np.zeros(k)
        self.perturbation = None
        self.all_constraints = []
        self.chosen_arms = np.zeros(self.T, dtype=int)
        self.do_attacks = np.zeros(self.T, dtype=int)

    def select_arm(self, t):
        # exploration phase
        if t < self.k:
          return t, False
        
        upper_conf = self.empirical_rewards + np.sqrt((2 * np.log(t)) / self.N)

        if self.target_arm:
          if upper_conf[self.target_arm] > upper_conf[0]:
            return self.target_arm, False
          return self.target_arm, True
        
        else:
          for j in range(1, self.k):
              if upper_conf[j] > upper_conf[0]:
                  best_arm = np.argmax(upper_conf[1:]) + 1
                  return best_arm, False
          best_arm = np.argmax(upper_conf[1:]) + 1
          return best_arm, True

    # do perturbation attack
    def find_perturbation(self, arm, t):
        x = cp.Variable(self.d)

        if self.reward_model is None:
          d_0 = self.empirical_means[arm] - self.empirical_means[0]
          c_0 = (math.sqrt((2 * math.log(t)) / self.N[0]) - math.sqrt((2 * math.log(t)) / self.N[arm])) - np.dot(self.true_means[0], d_0)
        else:
          d_0 = self.empirical_grad[arm] - self.empirical_grad[0]
          c_0 = (math.sqrt((2 * math.log(t)) / self.N[0]) - math.sqrt((2 * math.log(t)) / self.N[arm])) + (self.empirical_f[0] - self.empirical_f[arm])
        self.all_constraints.append((d_0, c_0))

        constraints = []
        for (d_0, c_0) in self.all_constraints:
            constraints.append(x @ d_0 >= c_0 + 1e-6)

        # face as feasbility problem
        if self.qp:
          objective = cp.Minimize(cp.norm(x, 2))
          prob = cp.Problem(objective, constraints)
        else:
          constraints.append(cp.norm(x, 2) <= self.epsilon)
          prob = cp.Problem(cp.Minimize(0), constraints)

        try:
          prob.solve(verbose=False)
        except cp.error.SolverError:
          return None

        if prob.status == 'optimal':
          return x.value
        else:
          print("don't find the perturbation")
          return None # can't find the optimal answer
        
    def update(self, arm, t, do_attack):
      # update based on new sample 
      sample = self.logged_data[arm][int(self.N[arm])]

      if self.real_data:
        image = self.encoder_preprocess(Image.open(sample)).unsqueeze(0).to('cuda')
        with torch.no_grad():
          sample = self.encoder_model.encode_image(image).view(-1)
          sample = sample.cpu()


      self.data[arm].append(sample) # store the data
      self.N[arm] += 1
      self.chosen_arms[t] = arm
      self.do_attacks[t] = do_attack

      if self.reward_model is not None:
        f_x, grad_x = fw0_and_grad(self.reward_model, torch.tensor(sample, dtype=torch.float32, device='cuda'))
        grad_x = grad_x.detach().cpu().numpy()

        self.empirical_f[arm] = self.empirical_f[arm] + (f_x - self.empirical_f[arm])/self.N[arm]
        self.empirical_grad[arm] = self.empirical_grad[arm] + (grad_x - self.empirical_grad[arm])/self.N[arm]

        if self.perturbation is not None:
          self.current_reward_model = load_params_to_new_model(self.reward_model, self.param_flat + torch.tensor(self.perturbation, device='cuda'))
      else:
        self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]


      # update based on last perturbation
      for j in range(self.k):
        if self.reward_model is None:
          self.empirical_rewards[j] = np.dot(self.true_means[0] + (self.perturbation if self.perturbation is not None else np.zeros(self.d)), self.empirical_means[j])
        else:
          if len(self.data[j]) > 0:
            if not self.real_data:
              self.empirical_rewards[j] = self.current_reward_model(torch.from_numpy(np.array(self.data[j], dtype=np.float32)).to(device='cuda')).mean().item()
            else:
              self.empirical_rewards[j] = self.current_reward_model(torch.from_numpy(np.array(self.data[j], dtype=np.float32)).to(device='cuda')).mean().item()

    def run(self):
        for t in tqdm(range(self.T)):
            # # select arm 
            arm, do_attack = self.select_arm(t)
            # print(f"step {t}: arm {arm} is selected and attack: {do_attack}")

            # attack part
            if t >= self.k and do_attack:
              status = self.find_perturbation(arm, t)

              if status is None:
                 return self.chosen_arms, self.do_attacks, self.perturbation
              else:
                 self.perturbation = status

            # update results
            self.update(arm, t, do_attack)

        return self.chosen_arms, self.do_attacks, self.perturbation
     
class FindPerturbationETC:
    def __init__(self, k, m, d, target_arm, true_means, logged_data, epsilon, qp=False, reward_model=None):
        self.k = k
        self.m = m
        self.d = d
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = epsilon
        self.qp = qp
        self.target_arm = target_arm
        self.reward_model = reward_model

        if self.reward_model is not None:
          self.param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
          w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
          print(f"Number of all params of the network is: {w}")
          d = w
          self.d = w
          self.empirical_f = np.zeros(k)
          self.empirical_grad = np.zeros((k, d))
          self.current_reward_model = copy.deepcopy(self.reward_model)

        self.N = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.perturbation = None

    def select_arm(self, t):
        if t < self.k * self.m:
            return t % self.k

        return self.target_arm

    def find_perturbation_with_l2_ball(self, arm, t):
        x = cp.Variable(self.d)
        constraints = []

        for j in range(self.k):
            if j != arm:
              if self.reward_model is None:
                d_j = self.empirical_means[arm] - self.empirical_means[j]
                c_j = - np.dot(self.true_means[0], d_j)
              else:
                d_j = self.empirical_grad[arm] - self.empirical_grad[j]
                c_j = - (self.empirical_f[arm] - self.empirical_f[j])

              constraints.append(x @ d_j >= c_j + 1e-6)

        if self.qp:
          objective = cp.Minimize(cp.norm(x, 2))
          prob = cp.Problem(objective, constraints)
        else:
          constraints.append(cp.norm(x, 2) <= self.epsilon)
          prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()

        if prob.status == 'optimal':
          return x.value
        else:
          return None


    def run(self, T):
        chosen_arms = np.zeros(T, dtype=int)

        for t in tqdm(range(T)):
            arm = self.select_arm(t)

            if t == self.m * self.k:
              self.perturbation = self.find_perturbation_with_l2_ball(arm, t)
              return chosen_arms

            sample = self.logged_data[arm][int(self.N[arm])]
            self.N[arm] += 1

            if self.reward_model is not None:
              f_x, grad_x = fw0_and_grad(self.reward_model, torch.tensor(sample, dtype=torch.float32, device='cuda'))
              grad_x = grad_x.detach().cpu().numpy()
              self.empirical_f[arm] = self.empirical_f[arm] + (f_x - self.empirical_f[arm])/self.N[arm]
              self.empirical_grad[arm] = self.empirical_grad[arm] + (grad_x - self.empirical_grad[arm])/self.N[arm]
            else:
              self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]

            chosen_arms[t] = arm

        return chosen_arms




class OSAEpsilonGreedy:
    def __init__(self, k, d, T, true_means, logged_data, epsilon_attack, qp=False, reward_model=None):
        self.k = k
        self.d = d
        self.T = T
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon_attack = epsilon_attack
        self.qp = qp
        self.epsilon = 0.1
        self.epsilon_min = 0.01


        self.reward_model = reward_model
        if self.reward_model is not None:
          self.param_flat = torch.cat([p.view(-1) for p in reward_model.parameters()])
          w = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
          print(f"Number of all params of the network is: {w}")
          d = w
          self.d = w
          self.empirical_f = np.zeros(k)
          self.empirical_grad = np.zeros((k, d))
          self.current_reward_model = copy.deepcopy(self.reward_model)

        self.data = [[] for _ in range(k)]
        self.N = np.zeros(k)
        self.empirical_means = np.zeros((k, d))
        self.empirical_rewards = np.zeros(k)
        self.perturbation = None
        self.chosen_arms = np.zeros(self.T, dtype=int)
        self.all_constraints = []
        self.do_attacks = np.zeros(self.T, dtype=int)


    def select_arm(self, t):
        if t < self.k:
            return t, False

        else:
          random.seed(42)
          if random.random() < self.epsilon:
            return random.randint(0, self.k - 1), False
          else:
            for j in range(1, self.k):
                if self.empirical_rewards[j] > self.empirical_rewards[0]:
                    best_arm = np.argmax(self.empirical_rewards[1:]) + 1
                    return best_arm, False
            best_arm = np.argmax(self.empirical_rewards[1:]) + 1
            return best_arm, True

    def find_perturbation(self, arm, t):
        x = cp.Variable(self.d)

        if self.reward_model is None:
          d_0 = self.empirical_means[arm] - self.empirical_means[0]
          c_0 = - np.dot(self.true_means[0], d_0)
        else:
          d_0 = self.empirical_grad[arm] - self.empirical_grad[0]
          c_0 = self.empirical_f[0] - self.empirical_f[arm]
        self.all_constraints.append((d_0, c_0))
        

        constraints = []
        for (d_0, c_0) in self.all_constraints:
            constraints.append(x @ d_0 >= c_0 + 1e-6)

        if self.qp:
          objective = cp.Minimize(cp.norm(x, 2))
          prob = cp.Problem(objective, constraints)
        else:
          constraints.append(cp.norm(x, 2) <= self.epsilon_attack)
          prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()

        if prob.status == 'optimal':
          return x.value
        else:
          print("I can't find the perturbation")
          return None


    def update(self, arm, t, do_attack):
      sample = self.logged_data[arm][int(self.N[arm])]
      self.data[arm].append(sample) # store the data
      self.N[arm] += 1
      self.chosen_arms[t] = arm
      self.do_attacks[t] = do_attack

      if self.reward_model is not None:
        f_x, grad_x = fw0_and_grad(self.reward_model, torch.tensor(sample, dtype=torch.float32, device='cuda'))
        grad_x = grad_x.detach().cpu().numpy()

        self.empirical_f[arm] = self.empirical_f[arm] + (f_x - self.empirical_f[arm])/self.N[arm]
        self.empirical_grad[arm] = self.empirical_grad[arm] + (grad_x - self.empirical_grad[arm])/self.N[arm]

        if self.perturbation is not None:
          self.current_reward_model = load_params_to_new_model(self.reward_model, self.param_flat + torch.tensor(self.perturbation, device='cuda'))
      else:
        self.empirical_means[arm] = self.empirical_means[arm] + (sample - self.empirical_means[arm])/self.N[arm]


      # update based on last perturbation
      for j in range(self.k):
        if self.reward_model is None:
          self.empirical_rewards[j] = np.dot(self.true_means[0] + (self.perturbation if self.perturbation is not None else np.zeros(self.d)), self.empirical_means[j])
        else:
          if len(self.data[j]) > 0:
            self.empirical_rewards[j] = self.current_reward_model(torch.from_numpy(np.array(self.data[j], dtype=np.float32)).to(device='cuda')).mean().item()


    def run(self):

        for t in tqdm(range(self.T)):
            arm, do_attack = self.select_arm(t)

            if t >= self.k and do_attack:
              status = self.find_perturbation(arm, t)
              if status is None:
                return self.chosen_arms, self.do_attacks
              self.perturbation = status

            self.update(arm, t, do_attack)


        return self.chosen_arms, self.do_attacks
    