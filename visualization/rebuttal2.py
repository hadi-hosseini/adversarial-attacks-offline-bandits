import matplotlib.pyplot as plt

# Corruption values
C = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]

# ---- Norm-2 Perturbation ----
pert_K3 = [0.230,0.281,0.280,0.256,0.284,0.330,0.318,0.342,0.342,0.342,0.241]
pert_K5 = [0.246,0.278,0.173,0.306,0.309,0.304,0.304,0.320,0.320,0.320,0.211]

# ---- Attack ----
attack_K3 = [0,0,0,0,0,1,1,1,1,1,1]
attack_K5 = [0,0,0,1,1,1,1,1,1,1,1]

# Colors
colors = ["#0072B2", "#D55E00"]

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,5))

# ----- Left subplot: Attack vs Corruption -----
ax1.step(C, attack_K3, where='mid', marker="o", color=colors[0], label="Attack K=3")
ax1.step(C, attack_K5, where='mid', marker="s", color=colors[1], label="Attack K=5")
ax1.set_xlabel(r"Corruption ($C$)")
ax1.set_ylabel("Attack (0 or 1)")
ax1.set_title("Attack vs Corruption")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# ----- Right subplot: Perturbation vs Corruption -----
ax2.plot(C, pert_K3, marker="o", color=colors[0], label="Perturbation K=3")
ax2.plot(C, pert_K5, marker="s", color=colors[1], label="Perturbation K=5")
ax2.set_xlabel(r"Corruption ($C$)")
ax2.set_ylabel("Norm-2 Perturbation")
ax2.set_title("Perturbation vs Corruption")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("Attack_and_Perturbation_vs_C.png", dpi=300)
plt.close()

print("Saved combined figure: Attack_and_Perturbation_vs_C.png")
