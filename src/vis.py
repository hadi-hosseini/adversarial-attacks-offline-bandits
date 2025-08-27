import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Make sure the directory exists
os.makedirs("visualization", exist_ok=True)

def plot_high_dim_vectors(v1, v2, dim=2, filename="vector_plot", data=None):
    """
    Project two high-dimensional vectors (and optional dataset) to 2D or 3D via PCA and plot them as arrows.

    Args:
        v1 (np.ndarray): First high-dim vector (e.g., w)
        v2 (np.ndarray): Second high-dim vector (e.g., w + delta)
        dim (int): 2 or 3 for 2D or 3D PCA visualization
        filename (str): Output file name (without extension)
        data (np.ndarray or None): Optional dataset (n_samples x d) to include and visualize as arrows
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    assert v1.shape == v2.shape, "Vectors must have the same dimension"
    assert dim in [2, 3], "dim must be 2 or 3"

    if data is not None:
        data = np.array(data)
        assert data.shape[1] == v1.shape[0], "data must have same feature dimension as vectors"
        data_for_pca = np.vstack([data, v1, v2])
    else:
        data_for_pca = np.vstack([v1, v2])

    # Perform PCA projection
    pca = PCA(n_components=dim)
    transformed = pca.fit_transform(data_for_pca)

    v1_t = transformed[-2]
    v2_t = transformed[-1]
    data_t = transformed[:-2] if data is not None else None

    if dim == 2:
        plt.figure()
        if data_t is not None:
            for i, vec in enumerate(data_t):
                plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                           color='green', alpha=1.0, label=f'μ{i+1}')
        plt.quiver(0, 0, v1_t[0], v1_t[1], angles='xy', scale_units='xy', scale=1,
                   color='red', label='w')
        plt.quiver(0, 0, v2_t[0], v2_t[1], angles='xy', scale_units='xy', scale=1,
                   color='blue', label='w + δ')
        all_x = [0, v1_t[0], v2_t[0]]
        all_y = [0, v1_t[1], v2_t[1]]
        if data_t is not None:
            all_x += list(data_t[:, 0])
            all_y += list(data_t[:, 1])
        plt.xlim(min(all_x) - 1, max(all_x) + 1)
        plt.ylim(min(all_y) - 1, max(all_y) + 1)
        plt.grid()
        plt.legend()
        plt.title("Vectors projected to 2D by PCA")
        plt.savefig(f"visualization/{filename}_2D.png")
        plt.close()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if data_t is not None:
            for i, vec in enumerate(data_t):
                ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='green', alpha=1.0, label=f'μ{i+1}')
        ax.quiver(0, 0, 0, v1_t[0], v1_t[1], v1_t[2], length=1.0, color='red', label='w')
        ax.quiver(0, 0, 0, v2_t[0], v2_t[1], v2_t[2], length=1.0, color='blue', label='w + δ')
        all_x = [0, v1_t[0], v2_t[0]]
        all_y = [0, v1_t[1], v2_t[1]]
        all_z = [0, v1_t[2], v2_t[2]]
        if data_t is not None:
            all_x += list(data_t[:, 0])
            all_y += list(data_t[:, 1])
            all_z += list(data_t[:, 2])
        ax.set_xlim([min(all_x) - 1, max(all_x) + 1])
        ax.set_ylim([min(all_y) - 1, max(all_y) + 1])
        ax.set_zlim([min(all_z) - 1, max(all_z) + 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Impact of Attack Vectors")
        ax.legend()
        plt.savefig(f"visualization/{filename}_3D.png")
        plt.close()


# # Example usage
# v1 = np.random.randn(1000)
# v2 = np.random.randn(1000)


# v1 = np.random.randn(1000)
# v2 = np.random.randn(1000)
# extra = np.random.randn(98, 1000)  # 98 extra vectors

# plot_two_high_dim_vectors(v1, v2, dim=2, filename="example_vectors")
# plot_two_high_dim_vectors(v1, v2, dim=3, filename="example_vectors", extra_samples=extra)
