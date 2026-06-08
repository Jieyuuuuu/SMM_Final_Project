"""
Ablation chart: the two intuitively-helpful tricks that turned out HARMFUL.

Compares, on the same DFA architecture:
  - DFA                         (no LayerNorm, no threshold calibration)  prefix DFA
  - DFA + LayerNorm             prefix AblLN
  - DFA + LayerNorm + thr-calib prefix AblLNcal

Parses TEST f1_macro mean+-std from the result .txt files (same data as
summarize_final.py) and writes ablation_bar.png.

Run from results/ with the project venv:  python plot_ablation.py
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
countries_display = ["China", "Iran", "UAE", "Cuba", "Russia", "Venezuela"]


def readf(prefix, c):
    for enc in ["utf-16", "utf-8", "gbk"]:
        try:
            t = open(os.path.join(HERE, f"zero-shot_{prefix}_{c}.txt"), encoding=enc).read()
            if t.strip():
                return t
        except Exception:
            pass
    return ""


def grab(text, tag):
    m = re.search(rf"\[{tag}\] f1_macro:\s*([0-9.]+)\+-([0-9.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"\[{tag}\] f1_macro:\s*([0-9.]+)", text)
    return (float(m.group(1)), 0.0) if m else (None, None)


def series(prefix):
    means, stds = [], []
    for c in countries:
        mu, sd = grab(readf(prefix, c), "TEST")
        means.append(mu if mu is not None else np.nan)
        stds.append(sd if sd is not None else 0.0)
    return means, stds


dfa_m, dfa_s = series("DFA")
ln_m, ln_s = series("AblLN")
lnc_m, lnc_s = series("AblLNcal")


def mean_of(v):
    arr = [x for x in v if not np.isnan(x)]
    return float(np.mean(arr))


def with_avg(v):
    return list(v) + [mean_of(v)]


dfa_avg, ln_avg, lnc_avg = mean_of(dfa_m), mean_of(ln_m), mean_of(lnc_m)

labels = countries_display + ["AVERAGE"]
x = np.arange(len(labels))
width = 0.26

C_DFA = '#2A9D8F'   # teal -- the ablation baseline (best)
C_LN = '#E63946'    # warning red -- harmful
C_LNC = '#F4A261'   # warning orange -- still harmful

dfa_b, ln_b, lnc_b = with_avg(dfa_m), with_avg(ln_m), with_avg(lnc_m)
dfa_be, ln_be, lnc_be = dfa_s + [0], ln_s + [0], lnc_s + [0]

fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
ekw = dict(ecolor='#333333', lw=1, capthick=1)
r1 = ax.bar(x - width, dfa_b, width, yerr=dfa_be, label='DFA (no LayerNorm, no calibration)',
            color=C_DFA, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)
r2 = ax.bar(x, ln_b, width, yerr=ln_be, label='+ LayerNorm  (harmful)',
            color=C_LN, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)
r3 = ax.bar(x + width, lnc_b, width, yerr=lnc_be, label='+ LayerNorm + threshold calibration  (harmful)',
            color=C_LNC, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)

ax.set_ylabel('Test F1-Macro Score', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Ablation — LayerNorm and Threshold Calibration both DEGRADE Performance',
             fontsize=15.5, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.axvline(x=len(countries_display) - 0.5, color='#999999', linestyle=':', linewidth=1.2)
# DFA baseline reference line (its average).
ax.axhline(y=dfa_avg, color=C_DFA, linestyle='--', linewidth=1.1, alpha=0.7)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0',
          fontsize=10.5, framealpha=0.95)

# Value labels on the AVERAGE group.
for rects in (r1, r2, r3):
    rect = rects[-1]
    ax.annotate(f'{rect.get_height():.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, color='#111111', weight='bold')

# Call out the dramatic collapses under LayerNorm (China & Cuba).
for cname in ("china", "cuba"):
    i = countries.index(cname)
    ax.annotate(f'{ln_m[i]:.3f}\ncollapse',
                xy=(x[i], ln_m[i]),
                xytext=(x[i], ln_m[i] - 0.22),
                ha='center', va='top', fontsize=8.5, color='#B5121B', weight='bold',
                arrowprops=dict(arrowstyle='->', color='#B5121B', lw=1.3))

plt.tight_layout()
out = os.path.join(HERE, "ablation_bar.png")
plt.savefig(out, bbox_inches='tight')
print(f"Saved {out}")
print(f"Averages -> DFA {dfa_avg:.4f} | +LN {ln_avg:.4f} | +LN+calib {lnc_avg:.4f}")
