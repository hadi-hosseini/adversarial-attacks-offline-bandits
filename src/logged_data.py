from scipy.stats import ortho_group
import numpy as np

# create offline dataset
def create_logged_data(k, d, n_samples, sigma, mu):
  samples = []
  for i in range(k):
    # Generate samples from N(μ_i, σ²I)
    samples.append(np.random.normal(loc=mu[i], scale=sigma, size=(n_samples, d)))
  return samples

# create orthogonal mus
def create_orthogonal_mu(k, d):
  orthogonal_matrix = ortho_group.rvs(dim=d)
  mu = orthogonal_matrix[:k]
  return mu

# verify orthogonality
def verify_mu(k, mu):
  for i in range(k):
      for j in range(i+1, k):
          print(f"Dot product μ_{i+1}·μ_{j+1}: {np.dot(mu[i], mu[j])}")

def create_bandit_instance(k, d, n_samples, sigma):
  mu = create_orthogonal_mu(k, d)
  logged_data = create_logged_data(k, d, n_samples, sigma, mu)
  return mu, logged_data

# Function to save mu and logged data to a file
def save_bandit_instance(mu, logged_data, filename):
    np.savez(filename, mu=mu, logged_data=logged_data)

# Function to load mu and logged data from a file
def load_bandit_instance(filename):
    data = np.load(filename, allow_pickle=True)
    mu = data['mu']
    logged_data = data['logged_data']
    return mu, logged_data

