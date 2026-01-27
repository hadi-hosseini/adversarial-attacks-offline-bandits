import matplotlib.pyplot as plt

# Epsilon values
eps = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

# ---- ASR Data ----
asr_100_K3 = [22.68,49.48,50.36,84.53,100,100,100,100,100,100]
asr_1000_K3 = [53.76,68.30,70.81,84.95,100,100,100,100,100,100]
asr_100_K5 = [34.73,42.10,96.84,100,100,100,100,100,100,100]
asr_1000_K5 = [10.65,67.73,99.69,100,100,100,100,100,100,100]

# ---- Perturbation Data ----
pert_100_K3 = [0.163,0.163,0.163,0.189,0.326,0.326,0.326,0.326,0.326,0.326]
pert_1000_K3 = [0.206,0.437,0.417,0.394,0.388,0.388,0.388,0.388,0.388,0.388]
pert_100_K5 = [0.149,0.155,0.198,0.313,0.313,0.313,0.313,0.313,0.313,0.313]
pert_1000_K5 = [0.158,0.375,0.317,0.405,0.405,0.405,0.405,0.405,0.405,0.405]

# Colors
colors = ["#0072B2", "#D55E00"]   # blue, orange

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,5))

# ----- Left subplot: ASR vs Epsilon -----
ax1.plot(eps, asr_100_K3, marker="o", color=colors[0], label="ASR (T=100, K=3)")
ax1.plot(eps, asr_1000_K3, marker="s", color=colors[1], label="ASR (T=1000, K=3)")
ax1.plot(eps, asr_100_K5, marker="^", color=colors[0], linestyle="--", label="ASR (T=100, K=5)")
ax1.plot(eps, asr_1000_K5, marker="v", color=colors[1], linestyle="--", label="ASR (T=1000, K=5)")
ax1.set_xlabel(r"$\varepsilon$-Contaminante")
ax1.set_ylabel("ASR (%)")
ax1.set_title("ASR vs Epsilon")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# ----- Right subplot: Perturbation vs Epsilon -----
ax2.plot(eps, pert_100_K3, marker="o", color=colors[0], label="Perturbation (T=100, K=3)")
ax2.plot(eps, pert_1000_K3, marker="s", color=colors[1], label="Perturbation (T=1000, K=3)")
ax2.plot(eps, pert_100_K5, marker="^", color=colors[0], linestyle="--", label="Perturbation (T=100, K=5)")
ax2.plot(eps, pert_1000_K5, marker="v", color=colors[1], linestyle="--", label="Perturbation (T=1000, K=5)")
ax2.set_xlabel(r"$\varepsilon$-Contaminante")
ax2.set_ylabel("Norm-2 Perturbation")
ax2.set_title("Perturbation vs Epsilon")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("ASR_and_Perturbation_vs_Epsilon.png", dpi=300)
plt.close()

print("Saved combined figure: ASR_and_Perturbation_vs_Epsilon.png")
