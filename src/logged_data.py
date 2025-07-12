from scipy.stats import ortho_group
import numpy as np

# create offline dataset
def create_logged_data(k, d, n_samples, sigma, mu):
  samples = []
  for i in range(k):
    # Generate samples from N(μ_i, σ²I)
    samples.append(np.random.normal(loc=mu[i], scale=sigma, size=(n_samples, d)))
  return samples

# create orthogonal mus (slow way)
def slow_create_orthogonal_mu(k, d):
  orthogonal_matrix = ortho_group.rvs(dim=d)
  mu = orthogonal_matrix[:k]
  return mu

# create orthogonal mus (fast way)
def fast_create_orthogonal_mu(k, d):
    A = np.random.randn(d, k)
    Q, _ = np.linalg.qr(A)  
    return Q.T 

# verify orthogonality
def verify_mu(k, mu):
  for i in range(k):
      for j in range(i+1, k):
        dot_product = np.dot(mu[i], mu[j])
        # print(f"Dot product μ_{i+1}·μ_{j+1}: {dot_product}")
        if abs(dot_product) >= 0.1:
            return False
  return True

def create_bandit_instance(k, d, n_samples, sigma):
  if d > 1000:
    mu = fast_create_orthogonal_mu(k, d)
    assert verify_mu(k, mu), "Mu vectors are not sufficiently orthogonal!"
  else:
    mu = slow_create_orthogonal_mu(k, d)
    assert verify_mu(k, mu), "Mu vectors are not sufficiently orthogonal!"

  logged_data = create_logged_data(k, d, n_samples, sigma, mu)
  return mu, logged_data

# Function to save mu and logged data to a file
def save_bandit_instance(mu, k, d, logged_data, filename):
    np.savez(filename, mu=mu, k=k, d=d, logged_data=logged_data)

# Function to load mu and logged data from a file
def load_bandit_instance(filename):
    data = np.load(filename, allow_pickle=True)
    mu = data['mu']
    logged_data = data['logged_data']
    return mu, logged_data

