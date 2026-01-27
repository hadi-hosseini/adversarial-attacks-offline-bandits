import numpy as np
from tqdm import tqdm
import math
import cvxpy as cp

from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance


def print_norms(x):
    norm_two = np.linalg.norm(x)
    norm_infinity = np.linalg.norm(x, ord=np.inf)
    print("norm 2:", norm_two)
    print("norm infinity: ", norm_infinity)

class OSA:
    def __init__(self, k, d, T, true_means, logged_data, epsilon, alpha, eps, qp=False, estimator="empirical"):
        self.k = k
        self.d = d
        self.T = T
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = epsilon
        self.counts = np.zeros(k, dtype=int)
        self.qp = qp
        self.data = [[] for _ in range(k)]
        self.perturbation = None
        self.sigma0 = 1.0
        self.alpha = alpha
        self.eps = eps
        self.all_constraints = []
        self.chosen_arms = np.zeros(self.T, dtype=int)
        self.do_attacks = np.zeros(self.T, dtype=int)
        self.number_of_selected_arm = np.zeros(k) 
        self.empirical_rewards = np.zeros(k)

        if estimator == "trimmed":
            self.estimator = lambda x: trimmed_mean_vectorize(x, true_means[0], alpha)
        elif estimator == "empirical":
            self.estimator = lambda x: np.zeros(d) if len(x) == 0 else np.mean(np.stack(x), axis=0)


    def select_arm(self, t):
        if t < self.k:
          return t, False
        
        indices = []
        for a in range(self.k):
            Na = self.counts[a]
            mu_hat = self.empirical_rewards[a]
            bonus = (self.sigma0 / (1 - 2*self.alpha)) * np.sqrt(4 * np.log(t) / Na)
            indices.append(mu_hat + bonus)

        for j in range(1, self.k):
            if indices[j] > indices[0]:
                best_arm = np.argmax(indices[1:]) + 1
                return best_arm, False
            
        #### should attack and check is possible and select the biggest one
        valid_arms = [j for j in np.arange(1, self.k) if self.number_of_selected_arm[j] < self.eps * self.counts[j]]

        if len(valid_arms) == 0:
            return 0, False
        else:
            # print(valid_arms)
            # print(indices)
            best_arm = 0
            best_val = -10000
            for j, val in enumerate(indices):
                if j == 0:
                    continue
                if val > best_val:
                    best_arm = j
                    best_val = val
            # print(best_arm)
            # print(best_val)
            return best_arm, True

        # best_arm = np.argmax(indices[1:]) + 1
        # return best_arm, True

    def find_perturbation(self, arm, t):
        self.number_of_selected_arm[arm] += 1
        x = cp.Variable(self.d)

        d_0 = self.estimator(self.data[arm]) - self.estimator(self.data[0])
        c_0 = (self.sigma0 / (1 - 2*self.alpha)) * ((math.sqrt((4 * math.log(t)) / self.counts[0]) - math.sqrt((4 * math.log(t)) / self.counts[arm]))) - np.dot(self.true_means[0], d_0)
        self.all_constraints.append((d_0, c_0))

        constraints = []
        for (d_0, c_0) in self.all_constraints:
            constraints.append(x @ d_0 >= c_0 + 1e-6)

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
        sample = self.logged_data[arm][self.counts[arm]]

        self.data[arm].append(sample)
        self.counts[arm] += 1

        self.chosen_arms[t] = arm
        self.do_attacks[t] = do_attack

        for j in range(self.k):
            self.empirical_rewards[j] = np.dot(self.true_means[0] + (self.perturbation if self.perturbation is not None else np.zeros(self.d)), self.estimator(self.data[j]))

    def run(self):
        for t in tqdm(range(self.T)):
            arm, do_attack = self.select_arm(t)

            if t >= self.k and do_attack:
              status = self.find_perturbation(arm, t)

              if status is None:
                 return self.chosen_arms, self.do_attacks, self.perturbation
              else:
                 self.perturbation = status

            self.update(arm, t, do_attack)

        return self.chosen_arms, self.do_attacks, self.perturbation




def trimmed_mean_vectorize(x, mu, alpha):
    x = np.asarray(x)
    n = x.shape[0]

    if n == 0:
        return np.zeros_like(mu)

    scores = np.dot(x, mu)

    sorted_idx = np.argsort(scores)
    x_sorted = x[sorted_idx]

    k = int(alpha * n)
    if n - 2*k > 0:
        trimmed_x = x_sorted[k:n-k]
    else:
        trimmed_x = x_sorted

    return np.mean(trimmed_x, axis=0)
    
def trimmed_mean(x, alpha):
    """
    α-trimmed mean: remove smallest α and largest α fraction of data.
    """
    n = len(x)
    if n == 0:
        return 0.0
    x_sorted = np.sort(x)
    k = int(alpha * n)
    return np.mean(x_sorted[k:n - k]) if n - 2*k > 0 else np.mean(x_sorted)

class crUCB:
    def __init__(self, K, T, alpha, true_means, logged_data, perturbation, estimator):
        self.K = K
        self.T = T
        self.alpha = alpha
        self.sigma0 = 1.0
        self.counts = np.zeros(K, dtype=int)
        self.rewards = [[] for _ in range(K)]
        self.estimator_name = estimator
        self.true_means = true_means
        self.logged_data = logged_data
        self.perturbation = perturbation
        
        if estimator == "trimmed":
            self.estimator = lambda x: trimmed_mean(x, alpha)
        elif estimator == "shorth":
            self.estimator = lambda x: shorth_mean(x, alpha)
        elif estimator == "empirical":
            self.estimator = lambda x: np.mean(x)

    def select_arm(self, t):
        if t < self.K:
            return t

        indices = []
        for a in range(self.K):
            Na = self.counts[a]
            mu_hat = self.estimator(self.rewards[a])
            bonus = (self.sigma0 / (1 - 2*self.alpha)) * np.sqrt(4 * np.log(t) / Na)
            indices.append(mu_hat + bonus)

        return np.argmax(indices)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm].append(reward)

    
    def get_reward(self, x):
        return np.dot(self.true_means[0] + self.perturbation, x)
    
    def run(self):
        chosen_arms = np.zeros(self.T, dtype=int)

        for t in tqdm(range(self.T)):
            arm = self.select_arm(t)
            sample = self.logged_data[arm][self.counts[arm]]
            reward = self.get_reward(sample)
            self.update(arm, reward)
            chosen_arms[t] = arm

        return chosen_arms

def osa(k, d, T, mu, alpha, eps, logged_data, epsilon_attack, qp=False):
    find_perturbation = OSA(k, d, T, mu, logged_data, epsilon_attack, alpha, eps, qp=qp, estimator='trimmed')
    chosen_arms, do_attacks, perturbation = find_perturbation.run()
    print("\nNumber of pulls per arm:", find_perturbation.counts)
    print(chosen_arms)
    print(do_attacks)
    print_norms(perturbation)

    crucb = crUCB(k, T, alpha, true_means, logged_data, perturbation, estimator="trimmed")
    chosen_arms = crucb.run()
    ASR = ((T - k + 1 - crucb.counts[0]) / (T - k)) * 100
    print(f"ASR: {ASR}")
    print("\nNumber of pulls per arm:", crucb.counts)
    print(chosen_arms)

if __name__ == "__main__":
    cfg = COFNIG()

    k = 5
    T = 1000
    d = 1000
    alpha = 0.1       
    sigma0 = 1.0  

    print(f"Bandit Instance with  T:{T}, K:{k}, D:{d}")
    print(60*'=')
    print(60*'=')

    if cfg.creat_new_instance:
        mu, logged_data = create_bandit_instance(k, d, cfg.n_samples, cfg.sigma)
        save_bandit_instance(mu, k, d, logged_data, cfg.bandit_instance_path)
        print("Save new Bandit Instance")
    else:
        mu, logged_data = load_bandit_instance(cfg.bandit_instance_path)
        print("Load Bandit Instance")

   
    true_means = mu
    perturbation = 0.0
    crucb = crUCB(k, T, alpha, true_means, logged_data, perturbation, estimator="trimmed") # ["empirical", "trimmed", "shorth"]
    chosen_arms = crucb.run()
    print("\nNumber of pulls per arm:", crucb.counts)
    print(chosen_arms)
    print(60*'=')
    print(60*'=')


    for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print("OSA Attack")
        print(f"EPSILON: {eps}")
        empirical_mus = mu
        osa(k, d, T, mu=mu, alpha=alpha, eps=eps, logged_data=logged_data, epsilon_attack=0.5, qp=False)
        print(60*'=')
        print(60*'=')



