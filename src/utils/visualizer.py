import matplotlib.pyplot as plt
import numpy as np

def generate_performance_plots(df_truth, dwell_logs, metrics_summary, output_file="benchmark_results.png"):
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1.2, 1]})

    # Find highest density 150ms window automatically
    bins = np.arange(0, df_truth['toa'].max(), 0.05)
    counts, edges = np.histogram(df_truth['toa'], bins=bins)
    peak_idx = int(np.argmax(counts))
    t_start_vis = max(0.0, edges[peak_idx] - 0.02)
    t_end_vis = t_start_vis + 0.15

    # 1. Trajectory vs Ground Truth
    ax0 = axes[0]
    sub_df = df_truth[(df_truth['toa'] >= t_start_vis) & (df_truth['toa'] <= t_end_vis)]
    
    ax0.scatter(
        (sub_df['toa'] - t_start_vis) * 1e3, sub_df['band_idx'], 
        c=sub_df['threat'], cmap='plasma', alpha=0.75, s=20, 
        label='Turing Truth Pulses (Threat Scaled)'
    )
    
    smart_dwells = dwell_logs['Smart Scan']
    zoom_dwells = smart_dwells[(smart_dwells['time'] >= t_start_vis) & (smart_dwells['time'] <= t_end_vis)]
    
    ax0.step(
        (zoom_dwells['time'] - t_start_vis) * 1e3, zoom_dwells['band'], 
        where='post', color='#00e5ff', linewidth=1.8, label='Adaptive RMAB Tuner Trajectory'
    )

    ax0.set_title(
        f"Turing Radar Environment & Adaptive ES Receiver Trajectory (Dense Window: {t_start_vis*1e3:.1f} - {t_end_vis*1e3:.1f} ms)", 
        fontsize=13, fontweight='bold'
    )
    ax0.set_xlabel("Relative Time in Window (ms)", fontsize=11)
    ax0.set_ylabel("Receiver Sub-band Index (0 - 15)", fontsize=11)
    ax0.set_xlim(0, 150)
    ax0.set_ylim(-0.5, 15.5)
    ax0.legend(loc='upper right', framealpha=0.9)

    # 2. Probability of Intercept
    ax1 = axes[1]
    strats = list(metrics_summary.keys())
    p_ints = [metrics_summary[s]['P_int'] for s in strats]
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    bars = ax1.bar(strats, p_ints, color=colors, width=0.45)
    ax1.set_title("Probability of Intercept (P_int %)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("P_int (%)", fontsize=11)
    ax1.set_ylim(0, max(max(p_ints) * 1.35, 25.0))
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.2f}%", ha='center', fontweight='bold')

    # 3. Detection Latency
    ax2 = axes[2]
    latencies = [metrics_summary[s]['Latency_ms'] for s in strats]
    bars2 = ax2.barh(strats, latencies, color=['#e67e22', '#7f8c8d', '#27ae60'], height=0.4)
    ax2.set_title("First-Intercept Latency (ms) — Lower is Better", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Latency (ms)", fontsize=11)
    for bar in bars2:
        xval = bar.get_width()
        ax2.text(xval + 10.0, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f} ms", va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n[Visualizer] Saved aligned diagnostic plot to: {output_file}")