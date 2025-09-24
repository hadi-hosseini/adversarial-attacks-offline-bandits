import numpy as np
import matplotlib.pyplot as plt

# --- ASR data for Random Reward Model ---
asr_data = {
    "T=100, K=3": np.array([
        93.81, 93.81, 98.96, 97.93, 97.93, 97.93, 98.96, 93.81, 100, 97.93,
        54.63, 88.65, 77.31, 93.81, 93.81, 97.93, 98.96, 96.90, 81.44, 98.96,
        50.51, 98.96, 95.87, 96.90, 98.96, 98.96, 95.87, 75.25, 98.96, 98.96
    ]),
    "T=100, K=4": np.array([
        100, 94.79, 97.91, 96.87, 98.95, 98.95, 97.91, 98.95, 100, 97.91,
        98.95, 98.95, 78.12, 93.75, 92.70, 95.83, 98.95, 98.95, 95.83, 98.95,
        92.70, 98.95, 98.95, 96.87, 98.95, 95.83, 98.95, 96.87, 100, 86.45
    ])
}

# Define bins
bins = np.linspace(0, 100, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Colors for bars
colors = ["#0072B2", "#D55E00"]

# Apply a clean style
plt.style.use("seaborn-v0_8-white")

plt.figure(figsize=(10, 6))
bar_width = (bins[1] - bins[0]) * 0.4  # width relative to bin size

# Plot grouped bars
for i, (label, values) in enumerate(asr_data.items()):
    counts, _ = np.histogram(values, bins=bins)
    prob = counts / counts.sum()
    plt.bar(
        bin_centers + (i - 0.5) * bar_width,
        prob,
        width=bar_width,
        color=colors[i],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        label=label
    )

# Labels and title
plt.xlabel("ASR", fontsize=14, fontweight="bold")
plt.ylabel("Probability", fontsize=14, fontweight="bold")
plt.title("Distribution of ASR values for Random Reward Model",
          fontsize=16, fontweight="bold")

# Ticks and subtle horizontal lines
plt.xticks(np.linspace(0, 100, 11), fontsize=12)
plt.yticks(np.linspace(0, 1, 11), fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

# Legend
plt.legend(frameon=True, fontsize=12)

plt.tight_layout()
plt.savefig("visualization/all_asr_random_model_distributions.png",
            dpi=300, bbox_inches="tight")
plt.show()
