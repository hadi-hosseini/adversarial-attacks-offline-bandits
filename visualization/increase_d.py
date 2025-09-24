import matplotlib.pyplot as plt
import numpy as np

# Data
dimensions = np.array([100, 500, 1000, 5000, 10000])

# Baseline results
norm_2_base = np.array([0.282, 0.185, 0.158, 0.097, 0.104])
norm_inf_base = np.array([0.068, 0.025, 0.018, 0.005, 0.004])

# QP results
norm_2_qp = np.array([0.221, 0.108, 0.117, 0.078, 0.086])
norm_inf_qp = np.array([0.055, 0.015, 0.013, 0.004, 0.003])

# colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


plt.figure(figsize=(8, 5))

# Baseline (red)
plt.plot(dimensions, norm_2_base, 'o-', color="#D55E00", label=r"Norm-2 (FP)")
plt.plot(dimensions, norm_inf_base, 's-', color="#D55E00", linestyle="--", label=r"Norm-$\infty$ (FP)")

# QP (blue)
plt.plot(dimensions, norm_2_qp, 'o-', color="#0072B2", label=r"Norm-2 (QP)")
plt.plot(dimensions, norm_inf_qp, 's-', color="#0072B2", linestyle="--", label=r"Norm-$\infty$ (QP)")

# Labels
plt.xlabel("Dimension (D)", fontsize=12)
plt.ylabel("Norm Value", fontsize=12)
plt.title("Norm-2 and Norm-$\infty$ Across Dimensions", fontsize=14)

# Grid, legend, layout
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Save & Show
plt.savefig("visualization/d_plot.png", dpi=300, bbox_inches="tight")
plt.show()
