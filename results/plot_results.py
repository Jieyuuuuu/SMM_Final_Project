import matplotlib.pyplot as plt
import numpy as np

# Set standard modern scientific plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
countries_display = ["China", "Iran", "UAE", "Cuba", "Russia", "Venezuela"]

# F1-Macro mean values
baseline_means = [0.5814, 0.7278, 0.8393, 0.8991, 0.7977, 0.9099]
dann_means = [0.5779, 0.7048, 0.8572, 0.7942, 0.8233, 0.7924]
coral_means = [0.5853, 0.7048, 0.8784, 0.8479, 0.8227, 0.7676]

# F1-Macro standard deviations
baseline_stds = [0.0589, 0.0143, 0.0593, 0.0535, 0.0193, 0.0107]
dann_stds = [0.0100, 0.0197, 0.0297, 0.1274, 0.0205, 0.0761]
coral_stds = [0.0305, 0.0155, 0.0158, 0.0790, 0.0299, 0.0305]

x = np.arange(len(countries))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

# Curated harmonious premium color palette
# Baseline: Light Slate Blue, DANN: Vibrant Amber, CORAL: Sleek Deep Violet (our champion!)
color_base = '#A0AAB2'  # Slate Gray/Blue
color_dann = '#FFAA00'  # Vibrant Amber
color_coral = '#7209B7' # Deep Violet Purple

# Plot bars with error bars (standard deviations)
rects1 = ax.bar(x - width, baseline_means, width, yerr=baseline_stds, 
                label='Paper Baseline (IOHunter)', color=color_base, 
                capsize=5, edgecolor='black', linewidth=0.8, error_kw=dict(ecolor='#333333', lw=1, capthick=1))

rects2 = ax.bar(x, dann_means, width, yerr=dann_stds, 
                label='DANN (Adversarial)', color=color_dann, 
                capsize=5, edgecolor='black', linewidth=0.8, error_kw=dict(ecolor='#333333', lw=1, capthick=1))

rects3 = ax.bar(x + width, coral_means, width, yerr=coral_stds, 
                label='Deep CORAL (Ours - Covariance)', color=color_coral, 
                capsize=5, edgecolor='black', linewidth=0.8, error_kw=dict(ecolor='#333333', lw=1, capthick=1))

# Add labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Test F1-Macro Score', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Cross-Country Zero-Shot Generalization Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(countries_display, fontsize=12, fontweight='bold')
ax.set_ylim(0.0, 1.05)

# Styling grid and ticks
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=11)

# Custom legend positioning and design
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=11, framealpha=0.95)

# Add values on top of bars (optional, only if F1 values are high and not cluttered)
def autolabel(rects, shift_x):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#111111', weight='bold')

autolabel(rects1, -width)
autolabel(rects2, 0)
autolabel(rects3, width)

# Optimize spacing
plt.tight_layout()

# Save final premium chart
chart_path = "zero-shot_comparison_chart.png"
plt.savefig(chart_path, bbox_inches='tight')
print(f"Chart successfully generated and saved as: {chart_path}")
