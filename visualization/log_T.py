import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Data
T = np.array([10, 100, 1000, 5000, 10000, 50000, 100000])
K3 = np.array([4, 6, 7, 8, 8, 9, 10])
K5 = np.array([1, 6, 8, 9, 10, 10, 11])

# colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


# Plot K=3 and K=5 vs T
plt.plot(T, K3, marker="o", color="#D55E00", label="K=3")
plt.plot(T, K5, marker="s", color="#0072B2", label="K=5")

# Plot log(T)
plt.plot(T, np.log(T), marker="^", linestyle="--", color="#009E73", label="log(T)")

# Labels and legend
plt.xlabel("Time Horizon (T)")
plt.ylabel("#Constraints")
plt.title("Number of Constraints in OSA Compared with log(T)")
plt.legend()
plt.grid(True)

# Set y-axis limit
plt.ylim(0, 20)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))


# Save the figure
plt.savefig("visualization/logT.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
