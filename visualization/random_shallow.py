import matplotlib.pyplot as plt

# Data for first plot - for different K values
hidden_size = [50, 150, 250, 500, 750, 1000]
asr_hidden_k3 = [13.0, 14.0, 22.0, 60.0, 100.0, 100.0]  
asr_hidden_k4 = [23, 71, 100, 100, 100, 100]  
asr_hidden_k5 = [13, 46, 51, 100, 100, 100]  

# Epsilon values
epsilon = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Data for second plot (different D values)
asr_d1000 = [14.14, 14.14, 14.14, 14.14, 14.14, 10.10, 10.10, 10.10, 11.11]
asr_d5000 = [16.16, 12.12, 11.11, 11.11, 10.10, 8.08, 8.08, 8.08, 8.08]
asr_d10000 = [12.12, 11.11, 11.11, 11.11, 11.11, 11.11, 10.10, 10.10, 7.07]

# Create figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


# First subplot: ASR vs Hidden Size for different K
axs[0].plot(hidden_size, asr_hidden_k3, marker='o', linestyle='-', linewidth=2,
            markersize=7, color='#D55E00', label="K = 3")
axs[0].plot(hidden_size, asr_hidden_k4, marker='s', linestyle='-', linewidth=2,
            markersize=7, color='#0072B2', label="K = 4")
axs[0].plot(hidden_size, asr_hidden_k5, marker='^', linestyle='-', linewidth=2,
            markersize=7, color='#009E73', label="K = 5")

axs[0].set_xlabel("Hidden Size", fontsize=12)
axs[0].set_ylabel("ASR (%)", fontsize=12)
axs[0].set_title("ASR Performance vs Hidden Size", fontsize=14)
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].set_ylim(0, 105)
axs[0].set_xticks(hidden_size)
axs[0].legend()

# Second subplot: ASR vs Epsilon for different D
axs[1].plot(epsilon, asr_d1000, marker='o', linestyle='-', linewidth=2,
            markersize=7, color='#D55E00', label="D = 1000")
axs[1].plot(epsilon, asr_d5000, marker='s', linestyle='-', linewidth=2,
            markersize=7, color='#0072B2', label="D = 5000")
axs[1].plot(epsilon, asr_d10000, marker='^', linestyle='-', linewidth=2,
            markersize=7, color='#009E73', label="D = 10000")

axs[1].set_xlabel("Epsilon", fontsize=12)
axs[1].set_ylabel("ASR (%)", fontsize=12)
axs[1].set_title("ASR Performance under Random Perturbation", fontsize=14)
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].set_xticks(epsilon)
axs[1].set_ylim(0, 35)
axs[1].legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig("visualization/combined_asr_plots.png", dpi=300)
plt.show()
