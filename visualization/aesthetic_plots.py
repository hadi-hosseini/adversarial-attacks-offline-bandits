import numpy as np
import matplotlib.pyplot as plt

# --- ASR data (without % sign) ---
asr_data = {
    "T=100, K=3": np.array([
        100, 100, 67.01, 100, 100, 100, 39.17, 87.62, 100, 90.72,
        77.31, 97.93, 93.81, 98.96, 95.87, 100, 81.44, 93.81, 98.96, 100,
        97.93, 100, 98.96, 91.75, 100, 100, 86.59, 100, 94.84, 89.69
    ]),
    "T=100, K=4": np.array([
        100, 100, 66.66, 100, 100, 100, 39.58, 100, 100, 91.66,
        77.08, 98.95, 93.75, 98.95, 95.83, 100, 72.91, 95.83, 100, 100,
        97.91, 100, 98.95, 91.66, 100, 100, 87.5, 98.95, 94.79, 89.58
    ])
}

# Define bins
bins = np.linspace(0, 100, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Colors for bars (colorblind-friendly palette)
colors = ["#0072B2", "#D55E00"]

# Apply a nice style
plt.style.use("seaborn-v0_8-white")

plt.figure(figsize=(10, 6))

bar_width = (bins[1] - bins[0]) * 0.4  # width relative to bin size

# Plot grouped bars with styling
for i, (label, values) in enumerate(asr_data.items()):
    counts, _ = np.histogram(values, bins=bins)
    prob = counts / counts.sum()
    plt.bar(
        bin_centers + (i - 0.5) * bar_width,
        prob,
        width=bar_width,
        color=colors[i],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
        label=label
    )

# Labels and title
plt.xlabel("ASR", fontsize=14, fontweight="bold")
plt.ylabel("Probability", fontsize=14, fontweight="bold")
plt.title("Distribution of ASR values for Aesthetic Reward Model",
          fontsize=16, fontweight="bold")

# Ticks and grid
plt.xticks(np.linspace(0, 100, 11), fontsize=12)
plt.yticks(np.linspace(0, 1, 11), fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Legend
plt.legend(frameon=True, fontsize=12)

plt.tight_layout()
plt.savefig("visualization/all_asr_aesthetic_model_distributions.png",
            dpi=300, bbox_inches="tight")
plt.show()
