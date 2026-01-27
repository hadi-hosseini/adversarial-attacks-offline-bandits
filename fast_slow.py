import numpy as np
from tqdm import tqdm
import math
import cvxpy as cp
import random

from src.config import COFNIG
from src.logged_data import create_bandit_instance, save_bandit_instance, load_bandit_instance

def print_norms(x):
    norm_two = np.linalg.norm(x)
    norm_infinity = np.linalg.norm(x, ord=np.inf)
    print("norm 2:", norm_two)
    print("norm infinity: ", norm_infinity)

class FastSlow:
    def __init__(self, K, T, C, true_means, logged_data, perturbation):
        self.K = K
        self.T = T
        self.C = C

        self.true_means = true_means
        self.logged_data = logged_data
        self.perturbation = perturbation
        self.N = { 'F': [0]*K, 'S': [0]*K }           
        self.empirical_reward = { 'F': [0.0]*K, 'S': [0.0]*K } 
        self.I = { 'F': set(), 'S': set() }        

        def _width(n, T):
            if n <= 0:
                return float('inf')
            return math.sqrt(math.log(T) / n) + (math.log(T) / n)
        self.width_fn = _width

    def _active_arms(self, l):
        return set(range(self.K)) - self.I[l]

    def _get_reward(self, x):
        return np.dot(self.true_means[0] + self.perturbation, x)

    def _choose_arm_for_instance(self, l):
        active = sorted(self._active_arms(l))
        if not active:
            return None
        counts = self.N[l]
        min_n = min(counts[a] for a in active)
        candidates = [a for a in active if counts[a] == min_n]
        return candidates[0]

    def _eliminate_if_possible(self, l):
        changed = True
        while changed:
            changed = False
            active = list(self._active_arms(l))
            widths = { a: self.width_fn(self.N[l][a], self.T) for a in active }
            mus = { a: self.empirical_reward[l][a] for a in active }

            for a in active:
                for aprime in active:
                    if a == aprime:
                        continue
                    if mus[a] - mus[aprime] > widths[a] + widths[aprime]:
                        self.I[l].add(aprime)
                        changed = True

                        if l == 'S':
                            self.I['F'].add(aprime)
                        break
                if changed:
                    break

    def run(self):
        t = 0
        while t < self.T:
            l = 'S' if random.random() < (1.0 / self.C) else 'F'

            if self._active_arms(l):
                arm = self._choose_arm_for_instance(l)
                if arm is None:
                    continue

                sample = self.logged_data[arm][self.N[l][arm]]
                r = self._get_reward(sample)
                self.empirical_reward[l][arm] = (self.N[l][arm] * self.empirical_reward[l][arm] + r) / (self.N[l][arm] + 1)
                self.N[l][arm] += 1
                t += 1
                self._eliminate_if_possible(l)

            else:
                other = 'S'
                other_active = sorted(self._active_arms(other))
                if not other_active:
                    break

                arm = random.choice(other_active)
                sample = self.logged_data[arm][self.N[l][arm]]
                r = self._get_reward(sample)
                self.empirical_reward[other][arm] = (self.N[other][arm] * self.empirical_reward[other][arm] + r) / (self.N[other][arm] + 1)
                self.N[other][arm] += 1
                t += 1

        return self.I


class OSA:
    def __init__(self, k, d, T, C, true_means, logged_data, epsilon, qp=False, target_arm=2):
        self.k = k
        self.d = d
        self.T = T
        self.C = C
        self.true_means = true_means
        self.logged_data = logged_data
        self.epsilon = epsilon
        self.qp = qp
        self.target_arm = target_arm
        self.perturbation = None
        self.all_constraints = []
        self.cost = 0.0
        
        self.N = { 'F': [0]*self.k, 'S': [0]*self.k }           
        self.empirical_reward = { 'F': [0.0]*self.k, 'S': [0.0]*self.k } 
        self.pure_reward = { 'F': [0.0]*self.k, 'S': [0.0]*self.k } 
        self.I = { 'F': set(), 'S': set() }  
        self.sum_data = {'F': [0.0]*self.k, 'S': [0.0]*self.k}

        def _width(n):
            if n <= 0:
                return float('inf')
            return math.sqrt(math.log(self.T) / n) + (math.log(self.T) / n)
        self.width_fn = _width

    def _active_arms(self, l):
        return set(range(self.k)) - self.I[l]

    def select_arm(self, l):
        active = sorted(self._active_arms(l))
        if not active:
            return None
        counts = self.N[l]
        min_n = min(counts[a] for a in active)
        candidates = [a for a in active if counts[a] == min_n]
        return candidates[0]

    def find_perturbation(self, arm, l):
        x = cp.Variable(self.d)

        d_0 = self.sum_data[l][arm]/self.N[l][arm] - self.sum_data[l][self.target_arm]/self.N[l][self.target_arm]
        c_0 = (self.width_fn(self.N[l][arm]) + self.width_fn(self.N[l][self.target_arm])) - np.dot(self.true_means[0], d_0)
        self.all_constraints.append((d_0, c_0))

        constraints = []
        for (d_0, c_0) in self.all_constraints:
            constraints.append(x @ d_0 <= c_0 + 1e-6)

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
        
    def update(self, arm, l):
        sample = self.logged_data[arm][self.N[l][arm]]
        self.sum_data[l][arm] += sample
        self.N[l][arm] += 1

        for j in range(self.k):
            if self.N[l][j] > 0:
                self.empirical_reward[l][j] = np.dot(self.true_means[0] + (self.perturbation if self.perturbation is not None else np.zeros(self.d)), self.sum_data[l][j] / self.N[l][j])
                self.pure_reward[l][j] = np.dot(self.true_means[0], self.sum_data[l][j] / self.N[l][j])
        
        self.cost = sum(abs(a - b) for a, b in zip(self.empirical_reward[l], self.pure_reward[l]))
        # print(self.cost)

    def _eliminate_if_possible(self, l):
        changed = True
        while changed:
            changed = False
            active = list(self._active_arms(l))
            widths = { a: self.width_fn(self.N[l][a]) for a in active }
            mus = { a: self.empirical_reward[l][a] for a in active }

            for a in active:
                for aprime in active:
                    if a == aprime:
                        continue

                    if (aprime == self.target_arm) and (mus[a] - mus[aprime] > widths[a] + widths[aprime]):
                        if self.cost < C:
                            # print("attack")
                            self.perturbation = self.find_perturbation(a, l)

                    elif mus[a] - mus[aprime] > widths[a] + widths[aprime]:
                        self.I[l].add(aprime)
                        changed = True

                        if l == 'S':
                            self.I['F'].add(aprime)
                        break
                if changed:
                    break

    def run(self):
        t = 0
        while t < self.T:
            l = 'S' if random.random() < (1.0 / self.C) else 'F'

            if self._active_arms(l):
                arm = self.select_arm(l)
                if arm is None:
                    continue

                self.update(arm, l)
                t += 1
                self._eliminate_if_possible(l)

            else:
                other = 'S'
                other_active = sorted(self._active_arms(other))
                if not other_active:
                    break

                arm = random.choice(other_active)
                self.update(arm, l)
                t += 1

        return self.I, self.perturbation

def osa(k, d, T, C, mu, logged_data, epsilon_attack, qp=False, target_arm=2):
    find_perturbation = OSA(k, d, T, C, mu, logged_data, epsilon_attack, qp=qp, target_arm=target_arm)
    results, perturbation = find_perturbation.run()
    print("Survivors (Fast):", sorted(set(range(k)) - results['F']))
    print("Survivors (Slow):", sorted(set(range(k)) - results['S']))
    if perturbation is None:
        perturbation = 0.0
        print("Perturbation: 0.0")
    else:
        print_norms(perturbation)
    print(60*'=')
    print(60*'=')

    print("Perturbed FAST&SLOW")
    fast_slow = FastSlow(k, T, C, mu, logged_data, perturbation)
    results = fast_slow.run()
    print("Survivors (Fast):", sorted(set(range(k)) - results['F']))
    print("Survivors (Slow):", sorted(set(range(k)) - results['S']))

    Attacked = 1 if target_arm in (set(range(k)) - results['S']) else 0
    print(f"Attacked: {Attacked}")

if __name__ == "__main__":
    cfg = COFNIG()

    k = 5
    T = 1000
    d = 1000
    target_arm = 2

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

    # print("FAST&SLOW")
    # perturbation = 0.0
    # fast_slow = FastSlow(k, T, C, mu, logged_data, perturbation)
    # result = fast_slow.run()

    # print("Survivors (Fast):", sorted(set(range(k)) - result['F']))
    # print("Survivors (Slow):", sorted(set(range(k)) - result['S']))

    # print(60*'=')
    # print(60*'=')


    for  C in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]:
        print(f"Corruption: {C}")
        print("OSA Attack")
        empirical_mus = mu
        osa(k, d, T, C, mu=mu, logged_data=logged_data, epsilon_attack=0.5, qp=False, target_arm=target_arm)
        print(60*'=')
        print(60*'=')