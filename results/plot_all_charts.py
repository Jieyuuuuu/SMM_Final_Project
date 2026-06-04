import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
countries_display = ["China", "Iran", "UAE", "Cuba", "Russia", "Venezuela"]

# F1-Macro statistics (Compiling ALL 4 models!)
baseline_means = [0.5814, 0.7278, 0.8393, 0.8991, 0.7977, 0.9099]
dann_means = [0.5779, 0.7048, 0.8572, 0.7942, 0.8233, 0.7924]
coral_means = [0.5853, 0.7048, 0.8784, 0.8479, 0.8227, 0.7676]
dfa_means = [0.6026, 0.7063, 0.8797, 0.8559, 0.8362, 0.7676]  # Our ultimate DFA-GFM!

baseline_stds = [0.0589, 0.0143, 0.0593, 0.0535, 0.0193, 0.0107]
dann_stds = [0.0100, 0.0197, 0.0297, 0.1274, 0.0205, 0.0761]
coral_stds = [0.0305, 0.0155, 0.0158, 0.0790, 0.0299, 0.0305]
dfa_stds = [0.0300, 0.0093, 0.0179, 0.0905, 0.0241, 0.0305]  # DFA-GFM stds (Notice Iran's 0.93% std!)

# Premium 4-Color Palette
color_base = '#A0AAB2'   # Slate Gray (Baseline)
color_dann = '#FFAA00'   # Amber Orange (DANN)
color_coral = '#7209B7'  # Royal Purple (CORAL)
color_dfa = '#06D6A0'    # Aurora Mint Green (Our DFA-GFM Champion!)

# ----------------------------------------------------
# CHART 1: 4-Way Grouped Bar Chart with Error Bars (Ultimate Comparison!)
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
x = np.arange(len(countries))
width = 0.20  # Reduced width to fit 4 bars perfectly

rects1 = ax.bar(x - 1.5*width, baseline_means, width, yerr=baseline_stds, 
                label='Paper Baseline (IOHunter)', color=color_base, 
                capsize=3, edgecolor='black', linewidth=0.5, error_kw=dict(ecolor='#333333', lw=0.6))

rects2 = ax.bar(x - 0.5*width, dann_means, width, yerr=dann_stds, 
                label='DANN (Adversarial)', color=color_dann, 
                capsize=3, edgecolor='black', linewidth=0.5, error_kw=dict(ecolor='#333333', lw=0.6))

rects3 = ax.bar(x + 0.5*width, coral_means, width, yerr=coral_stds, 
                label='Deep CORAL (Global)', color=color_coral, 
                capsize=3, edgecolor='black', linewidth=0.5, error_kw=dict(ecolor='#333333', lw=0.6))

rects4 = ax.bar(x + 1.5*width, dfa_means, width, yerr=dfa_stds, 
                label='DFA-GFM (Ours - Decoupled Alignment)', color=color_dfa, 
                capsize=3, edgecolor='black', linewidth=0.5, error_kw=dict(ecolor='#333333', lw=0.6))

ax.set_ylabel('Test F1-Macro Score', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Cross-Country Zero-Shot Generalization Comparison (F1-Macro)', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(countries_display, fontsize=11, fontweight='bold')
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc='lower left', frameon=True, edgecolor='#e0e0e0', fontsize=9, framealpha=0.95)

def autolabel(rects, target_ax=None, font_color='#222222'):
    curr_ax = target_ax if target_ax is not None else ax
    for rect in rects:
        height = rect.get_height()
        curr_ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color=font_color, weight='bold')

def autolabel_sub(rects, font_color='#222222'):
    autolabel(rects, ax, font_color)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4, font_color='#026c51') # Highlight our DFA labels in deep green

plt.tight_layout()
plt.savefig("zero-shot_comparison_bar.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# CHART 2: Trend Line Chart with Shaded Error Bands (4-Way)
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)

ax.plot(countries_display, baseline_means, marker='o', linestyle='-', linewidth=1.5, color=color_base, label='Paper Baseline')
ax.plot(countries_display, dann_means, marker='s', linestyle='--', linewidth=1.5, color=color_dann, label='DANN')
ax.plot(countries_display, coral_means, marker='D', linestyle=':', linewidth=1.5, color=color_coral, label='Deep CORAL')
ax.plot(countries_display, dfa_means, marker='^', linestyle='-', linewidth=2.5, color=color_dfa, label='DFA-GFM (Ours)')

# Shaded error bands
ax.fill_between(countries_display, np.array(baseline_means) - np.array(baseline_stds), 
                np.array(baseline_means) + np.array(baseline_stds), color=color_base, alpha=0.10)
ax.fill_between(countries_display, np.array(dann_means) - np.array(dann_stds), 
                np.array(dann_means) + np.array(dann_stds), color=color_dann, alpha=0.10)
ax.fill_between(countries_display, np.array(coral_means) - np.array(coral_stds), 
                np.array(coral_means) + np.array(coral_stds), color=color_coral, alpha=0.10)
ax.fill_between(countries_display, np.array(dfa_means) - np.array(dfa_stds), 
                np.array(dfa_means) + np.array(dfa_stds), color=color_dfa, alpha=0.20)

ax.set_ylabel('Test F1-Macro Score', fontsize=12, fontweight='bold')
ax.set_title('Generalization General Trends & Confidence Intervals (4-Way)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0.45, 1.0)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower left', frameon=True, edgecolor='#e0e0e0', fontsize=10)

plt.tight_layout()
plt.savefig("zero-shot_comparison_line.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# CHART 3: 4-Way Polar Radar / Spider Chart (Stunning Visuals!)
# ----------------------------------------------------
labels = countries_display
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

base_radar = baseline_means + baseline_means[:1]
dann_radar = dann_means + dann_means[:1]
coral_radar = coral_means + coral_means[:1]
dfa_radar = dfa_means + dfa_means[:1]

fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], labels, fontsize=11, fontweight='bold')

ax.set_rlabel_position(0)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["0.6", "0.7", "0.8", "0.9", "1.0"], color="grey", size=9)
plt.ylim(0.5, 1.0)

ax.plot(angles, base_radar, color=color_base, linewidth=1.2, linestyle='solid', label="Paper Baseline")
ax.fill(angles, base_radar, color=color_base, alpha=0.05)

ax.plot(angles, dann_radar, color=color_dann, linewidth=1.2, linestyle='dashed', label="DANN")
ax.fill(angles, dann_radar, color=color_dann, alpha=0.05)

ax.plot(angles, coral_radar, color=color_coral, linewidth=1.2, linestyle='dotted', label="Deep CORAL")
ax.fill(angles, coral_radar, color=color_coral, alpha=0.05)

ax.plot(angles, dfa_radar, color=color_dfa, linewidth=2.8, linestyle='solid', label="DFA-GFM (Ours)")
ax.fill(angles, dfa_radar, color=color_dfa, alpha=0.25)

ax.set_title("Zero-Shot Generalization Envelope Profile (4-Way)", fontsize=14, fontweight='bold', pad=25)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, edgecolor='#e0e0e0', fontsize=9)

plt.tight_layout()
plt.savefig("zero-shot_comparison_radar.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# CHART 4: Sub-Network Ablation on Russia (Updated with DFA!)
# ----------------------------------------------------
subnetworks = ["co-Retweet (coRT)", "co-URL (coURL)", "co-Hashtag (hashSeq)", "Fast Retweet (fastRT)", "Tweet Similarity (tweetSim)"]

sub_baseline = [0.7895, 0.8145, 0.4080, 0.4080, 0.4080]
sub_dann = [0.7905, 0.8214, 0.3832, 0.3832, 0.3832]     
sub_coral = [0.8147, 0.8252, 0.3834, 0.3950, 0.3834]    
sub_dfa = [0.8072, 0.8146, 0.4214, 0.4214, 0.4214]  # Russia DFA subnets (DFA successfully regularized structural paths)

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
x_sub = np.arange(len(subnetworks))
width_sub = 0.20

rects_sub1 = ax.bar(x_sub - 1.5*width_sub, sub_baseline, width_sub, label='Baseline', color=color_base, edgecolor='black', linewidth=0.4)
rects_sub2 = ax.bar(x_sub - 0.5*width_sub, sub_dann, width_sub, label='DANN', color=color_dann, edgecolor='black', linewidth=0.4)
rects_sub3 = ax.bar(x_sub + 0.5*width_sub, sub_coral, width_sub, label='Deep CORAL', color=color_coral, edgecolor='black', linewidth=0.4)
rects_sub4 = ax.bar(x_sub + 1.5*width_sub, sub_dfa, width_sub, label='DFA-GFM (Ours)', color=color_dfa, edgecolor='black', linewidth=0.4)

ax.set_ylabel('Sub-Network Test F1-Macro', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study: Similarity Sub-Networks Analysis (Russia)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_sub)
ax.set_xticklabels(subnetworks, fontsize=10, fontweight='bold')
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, edgecolor='#e0e0e0', fontsize=9)

autolabel_sub(rects_sub1)
autolabel_sub(rects_sub2)
autolabel_sub(rects_sub3)
autolabel_sub(rects_sub4)

plt.tight_layout()
plt.savefig("russia_subnetworks_ablation.png", bbox_inches='tight')
plt.close()

print("All premium 4-way scientific charts generated successfully!")
