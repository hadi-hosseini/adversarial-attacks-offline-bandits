import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Data from LaTeX table (mm:ss → seconds)
# -------------------------------


osa_data = [
    (3, 10000, 1, 3000, 1),
    (3, 1000, 0, 6000, 3),
    (5, 10000, 0, 3000, 4),
    (5, 1000, 0, 6000, 9),
]

trajfree_data = [
    (3, 10000, 25, 3000, 100),
    (3, 1000, 3, 6000, 245),
    (5, 10000, 23, 3000, 81),
    (5, 1000, 3, 6000, 201),
]

fulltraj_data = [
    (3, 10000, 91, 3000, 371),
    (3, 1000, 10, 6000, 814),
    (5, 10000, 225, 3000, 831),
    (5, 1000, 23, 6000, 1845),
]

# Attack methods
methods = ["OSA", "Trajectory-Free", "Full Trajectory"]
colors = ["#0072B2", "#D55E00", "#009E73"]

# Categories
linear_categories = [f"K={K}, d={d}" for (K, d, _, _, _) in osa_data]
nonlinear_categories = [f"K={K}, maxW={w}" for (K, _, _, w, _) in osa_data]

# Extract values
linear_values = [
    [x[2] for x in osa_data],
    [x[2] for x in trajfree_data],
    [x[2] for x in fulltraj_data],
]
nonlinear_values = [
    [x[4] for x in osa_data],
    [x[4] for x in trajfree_data],
    [x[4] for x in fulltraj_data],
]

# -------------------------------
# Combined Plot (Linear + Non-linear side by side)
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)



import matplotlib.ticker as mticker

def plot_attack_times(ax, values, categories, model_name):
    x = np.arange(len(categories))
    width = 0.25

    for i, (method, vals) in enumerate(zip(methods, values)):
        # Avoid log(0): shift zeros to tiny positive
        safe_vals = [v if v > 0 else 0.1 for v in vals]
        bars = ax.bar(x + i*width, safe_vals, width, label=method, color=colors[i])

        # Add real values on top of each bar
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,  # x position (center)
                height * 1.05,                    # y position (slightly above the bar)
                f"{val}",                         # text
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=12, rotation=25, fontweight="bold")

    # Y-axis log scale but labels in real seconds
    ax.set_yscale("log")
    ax.set_ylabel("Attack Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} Reward Model", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Format y-ticks nicely
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
# Left = Linear, Right = Non-linear
plot_attack_times(axes[0], linear_values, linear_categories, "Linear")
plot_attack_times(axes[1], nonlinear_values, nonlinear_categories, "Non-linear")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualization/combined_attack_times.png", dpi=300, bbox_inches="tight")
plt.show()
