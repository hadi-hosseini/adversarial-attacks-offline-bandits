import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Create visualization directory if it doesn't exist
os.makedirs("visualization", exist_ok=True)

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa

# Make sure the directory exists
os.makedirs("visualization", exist_ok=True)

def plot_two_high_dim_vectors(v1, v2, dim=2, filename="vector_plot", extra_samples=None):
    v1 = np.array(v1)
    v2 = np.array(v2)
    assert v1.shape == v2.shape, "Vectors must have the same dimension"
    assert dim in [2, 3], "dim must be 2 or 3"

    # Combine vectors with extra_samples (if any) for PCA fitting
    if extra_samples is not None:
        extra_samples = np.array(extra_samples)
        assert extra_samples.shape[1] == v1.shape[0], "extra_samples must have same feature dimension"
        data_for_pca = np.vstack([v1, v2, extra_samples])
    else:
        data_for_pca = np.vstack([v1, v2])

    # Fit PCA on combined data
    pca = PCA(n_components=dim)
    pca.fit(data_for_pca)
    v1_t = pca.transform(v1.reshape(1, -1))[0]
    v2_t = pca.transform(v2.reshape(1, -1))[0]

    if dim == 2:
        plt.figure()
        plt.quiver(0, 0, v1_t[0], v1_t[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
        plt.quiver(0, 0, v2_t[0], v2_t[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
        all_x = [0, v1_t[0], v2_t[0]]
        all_y = [0, v1_t[1], v2_t[1]]
        plt.xlim(min(all_x) - 1, max(all_x) + 1)
        plt.ylim(min(all_y) - 1, max(all_y) + 1)
        plt.grid()
        plt.legend()
        plt.title("Two vectors projected to 2D by PCA")
        plt.savefig(f"visualization/{filename}_2D.png")
        plt.close()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, v1_t[0], v1_t[1], v1_t[2], length=1, color='r', label='v1')
        ax.quiver(0, 0, 0, v2_t[0], v2_t[1], v2_t[2], length=1, color='b', label='v2')
        all_x = [0, v1_t[0], v2_t[0]]
        all_y = [0, v1_t[1], v2_t[1]]
        all_z = [0, v1_t[2], v2_t[2]]
        ax.set_xlim([min(all_x) - 1, max(all_x) + 1])
        ax.set_ylim([min(all_y) - 1, max(all_y) + 1])
        ax.set_zlim([min(all_z) - 1, max(all_z) + 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.title("Two vectors projected to 3D by PCA")
        plt.savefig(f"visualization/{filename}_3D.png")
        plt.close()


# Example usage
v1 = np.random.randn(1000)
v2 = np.random.randn(1000)


v1 = np.random.randn(1000)
v2 = np.random.randn(1000)
extra = np.random.randn(98, 1000)  # 98 extra vectors

plot_two_high_dim_vectors(v1, v2, dim=2, filename="example_vectors")
plot_two_high_dim_vectors(v1, v2, dim=3, filename="example_vectors", extra_samples=extra)
