import numpy as np
import matplotlib.pyplot as plt

# --- ASR data for Image Reward Model ---
asr_data = {
    "T=100, K=3": np.array([
        98.96, 94.84, 100, 93.81, 100, 96.90, 98.96, 98.96, 10.30, 100,
        3.09, 25.77, 100, 100, 100, 100, 95.87, 100, 20.61, 100,
        85.56, 100, 100, 97.93, 20.61, 37.11, 100, 100, 59.79, 100
    ]),
    "T=100, K=4": np.array([
        98.95, 96.87, 100, 98.95, 100, 4.16, 100, 98.95, 11.45, 100,
        3.12, 27.08, 100, 100, 100, 100, 95.83, 100, 44.79, 100,
        96.87, 100, 100, 97.91, 100, 98.95, 100, 100, 62.5, 100
    ])
}

# Define bins
bins = np.linspace(0, 100, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Colors for bars (colorblind-friendly)
colors = ["#0072B2", "#D55E00"]

plt.style.use("seaborn-v0_8-white")  # clean style without grid

plt.figure(figsize=(10, 6))

bar_width = (bins[1] - bins[0]) * 0.4  # width relative to bin size

# Compute histograms and plot grouped bars
for i, (label, values) in enumerate(asr_data.items()):
    counts, _ = np.histogram(values, bins=bins)
    prob = counts / counts.sum()  # normalize to probability
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
plt.title("Distribution of ASR values for Image Reward Model",
          fontsize=16, fontweight="bold")

# Ticks only, no grid
plt.xticks(np.linspace(0, 100, 11), fontsize=12)
plt.yticks(np.linspace(0, 1, 11), fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Legend
plt.legend(frameon=True, fontsize=12)

plt.tight_layout()
plt.savefig("visualization/all_asr_image_reward_distributions.png",
            dpi=300, bbox_inches="tight")
plt.show()
