import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
attack_methods = ["OSA", "Trajectory-Free", "Full Trajectory"]
K_values = [4, 3]

asr_data = {
    "OSA": {4: (100, 30.20), 3: (100, 10.30)},
    "Trajectory-Free": {4: (100, 26.04), 3: (100, 6.18)},
    "Full Trajectory": {4: (100, 81.25), 3: (100, 63.91)}
}

# --- Colors ---
colors = ["#0072B2", "#D55E00", "#009E73"]

bar_width = 0.1      # width of each bar
method_gap = 0.05    # gap between different methods
group_gap = 0.2      # gap between K groups

# Base x positions for K groups
x = np.arange(len(K_values)) * (len(attack_methods) * (2*bar_width + method_gap) + group_gap)

fig, ax = plt.subplots(figsize=(9,5))

# --- Plot bars ---
for i, method in enumerate(attack_methods):
    before = [asr_data[method][k][0] for k in K_values]
    after = [asr_data[method][k][1] for k in K_values]

    offset = i * (2*bar_width + method_gap)
    ax.bar(x + offset, before, width=bar_width, color=colors[i], alpha=0.8, label=f"{method} (Before)")
    ax.bar(x + offset + bar_width, after, width=bar_width, color=colors[i], alpha=0.4, label=f"{method} (After)")

# --- Formatting ---
ax.set_xticks(x + (len(attack_methods)-1)*(2*bar_width+method_gap)/2 + bar_width/2)
ax.set_xticklabels([f"K={k}" for k in K_values])
ax.set_ylabel("ASR")
ax.set_title("ASR Before and After Defense")

# --- Legend (remove duplicates) ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
leg = ax.legend(by_label.values(), by_label.keys(),
          loc='upper right', frameon=True)
leg.get_frame().set_alpha(0.5)


# --- Layout & Save ---
plt.tight_layout()
plt.savefig("visualization/dfense_t4.png", dpi=300)
plt.show()
