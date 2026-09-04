import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

splits = ['test_0', 'test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6']
rr_pint = [4.54, 3.96, 4.37, 4.37, 4.39, 4.38, 4.37]
smart_pint = [41.84, 64.00, 85.66, 56.31, 65.18, 87.41, 93.31]
rr_lat = [755.3, 620.2, 885.7, 844.3, 828.7, 965.8, 1413.9]
smart_lat = [713.5, 523.2, 645.0, 267.4, 419.4, 960.4, 961.5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# Plot 1: Intercept Probability
x = np.arange(len(splits))
w = 0.35

ax1.bar(x - w/2, rr_pint, width=w, label='Round-Robin Sweep (Baseline)', color='#e74c3c')
ax1.bar(x + w/2, smart_pint, width=w, label='Adaptive Bayesian RMAB (Ours)', color='#2ecc71')
ax1.set_xticks(x)
ax1.set_xticklabels(splits, fontweight='bold')
ax1.set_ylabel('Probability of Intercept (%)', fontsize=11, fontweight='bold')
ax1.set_title('Probability of Intercept (P_int) Across 7 Official Turing Splits', fontsize=13, fontweight='bold')
ax1.axhline(np.mean(smart_pint), color='#27ae60', linestyle='--', alpha=0.8, label=f'Mean Ours: {np.mean(smart_pint):.1f}%')
ax1.legend(frameon=True, facecolor='white')

# Plot 2: Latency Reduction
ax2.bar(x - w/2, rr_lat, width=w, label='Round-Robin Sweep', color='#e67e22')
ax2.bar(x + w/2, smart_lat, width=w, label='Adaptive Bayesian RMAB', color='#3498db')
ax2.set_xticks(x)
ax2.set_xticklabels(splits, fontweight='bold')
ax2.set_ylabel('First-Intercept Latency (ms)', fontsize=11, fontweight='bold')
ax2.set_title('First-Intercept Latency (Lower is Better)', fontsize=13, fontweight='bold')
ax2.axhline(np.mean(smart_lat), color='#2980b9', linestyle='--', alpha=0.8, label=f'Mean Ours: {np.mean(smart_lat):.1f} ms')
ax2.legend(frameon=True, facecolor='white')

plt.tight_layout()
Path("assets").mkdir(exist_ok=True)
plt.savefig("assets/cross_validation_rigor.png", dpi=300)
print("Updated asset generated: assets/cross_validation_rigor.png")