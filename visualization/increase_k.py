import matplotlib.pyplot as plt
import numpy as np

# Data
k_values = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

norm_2 = np.array([0.2964, 0.3117, 0.2978, 0.3159, 0.3020, 0.3060,
                   0.3196, 0.2911, 0.2959, 0.2892, 0.2640, 0.2635,
                   0.2686, 0.2668, 0.2501, 0.2437, 0.2453, 0.2281,
                   0.2129, 0.2217])

norm_inf = np.array([0.0343, 0.0364, 0.0304, 0.0325, 0.0351, 0.0306,
                     0.0401, 0.0328, 0.0294, 0.0289, 0.0320, 0.0277,
                     0.0275, 0.0278, 0.0285, 0.0297, 0.0237, 0.0227,
                     0.0234, 0.0254])

# Plot
plt.figure(figsize=(8, 5))

plt.plot(k_values, norm_2, 'o-', color="red", label="Norm-2") # #d62728
plt.plot(k_values, norm_inf, 's--', color="blue", label="Norm-∞") #  #1f77b4

# Labels
plt.xlabel("K (Number of Arms)", fontsize=12)
plt.ylabel("Norm Value", fontsize=12)
plt.title("Norm-2 and Norm-∞ Across K", fontsize=14)

# Grid, legend, layout
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Save & Show
plt.savefig("visualization/k_plot.png", dpi=300)
plt.show()
