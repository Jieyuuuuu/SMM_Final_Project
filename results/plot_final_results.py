"""
Report-ready bar charts for the final IOHunter+ zero-shot results.

Parses TEST f1_macro mean+-std from the result .txt files (same data as
summarize_final.py) and produces two PNGs:
  - final_results_bar.png : grouped per-country comparison + AVERAGE group
  - final_results_avg.png : average-per-method headline bar

Run from results/ with the project venv:  python plot_final_results.py
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

# External paper numbers (Table 3 Only-PreTrain) -- NOT reproducible here.
paper_mean = {"china":0.5814,"iran":0.7278,"UAE":0.8393,"cuba":0.8991,"russia":0.7977,"venezuela":0.9099}
paper_std  = {"china":0.0589,"iran":0.0143,"UAE":0.0593,"cuba":0.0535,"russia":0.0193,"venezuela":0.0107}

# Candidate lambdas for the adaptive method: lambda -> file prefix.
ADAPTIVE = {0.0: "CW000", 0.1: "CW010", 0.3: "CW030", 1.0: "DFA"}


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
    """Return (mean, std) for the given metric tag, or (None, None)."""
    m = re.search(rf"\[{tag}\] f1_macro:\s*([0-9.]+)\+-([0-9.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"\[{tag}\] f1_macro:\s*([0-9.]+)", text)
    return (float(m.group(1)), 0.0) if m else (None, None)


def method_series(prefix):
    means, stds = [], []
    for c in countries:
        mu, sd = grab(readf(prefix, c), "TEST")
        means.append(mu if mu is not None else np.nan)
        stds.append(sd if sd is not None else 0.0)
    return means, stds


def adaptive_series():
    means, stds, lams = [], [], []
    for c in countries:
        best = None  # (val_mean, test_mean, test_std, lam)
        for lam, prefix in ADAPTIVE.items():
            t = readf(prefix, c)
            v_mu, _ = grab(t, "VAL")
            te_mu, te_sd = grab(t, "TEST")
            if v_mu is None or te_mu is None:
                continue
            if best is None or v_mu > best[0]:
                best = (v_mu, te_mu, te_sd, lam)
        means.append(best[1]); stds.append(best[2]); lams.append(best[3])
    return means, stds, lams


# ---- Gather data ----------------------------------------------------------
paper_m = [paper_mean[c] for c in countries]
paper_s = [paper_std[c] for c in countries]
repro_m, repro_s = method_series("CW000")
dfa_m, dfa_s = method_series("DFA")
adapt_m, adapt_s, adapt_lams = adaptive_series()
dann_m, _ = method_series("DANN")
coral_m, _ = method_series("CORAL")

# Colours
C_PAPER = '#C9CCD1'   # light grey (external)
C_REPRO = '#A0AAB2'   # slate
C_DFA   = '#FFAA00'   # amber
C_OURS  = '#7209B7'   # deep violet -- champion

# ===========================================================================
# Chart 1: grouped per-country bars + AVERAGE group
# ===========================================================================
labels = countries_display + ["AVERAGE"]
x = np.arange(len(labels))
width = 0.20

def with_avg(vals):
    arr = [v for v in vals if not np.isnan(v)]
    return list(vals) + [np.mean(arr)]

paper_b = with_avg(paper_m)
repro_b = with_avg(repro_m)
dfa_b = with_avg(dfa_m)
ours_b = with_avg(adapt_m)
# No error bar on the AVERAGE bar (it's a mean of means).
paper_be = paper_s + [0]; repro_be = repro_s + [0]; dfa_be = dfa_s + [0]; ours_be = adapt_s + [0]

fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
ekw = dict(ecolor='#333333', lw=1, capthick=1)
r1 = ax.bar(x - 1.5*width, paper_b, width, yerr=paper_be, label='Paper (external, not reproducible)',
            color=C_PAPER, hatch='//', capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)
r2 = ax.bar(x - 0.5*width, repro_b, width, yerr=repro_be, label='no-DA reproducible baseline (λ=0)',
            color=C_REPRO, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)
r3 = ax.bar(x + 0.5*width, dfa_b, width, yerr=dfa_be, label='DFA-GFM (fixed λ=1.0)',
            color=C_DFA, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)
r4 = ax.bar(x + 1.5*width, ours_b, width, yerr=ours_be, label='CAL-DFA-Adaptive (Ours)',
            color=C_OURS, capsize=4, edgecolor='black', linewidth=0.8, error_kw=ekw)

ax.set_ylabel('Test F1-Macro Score', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Zero-Shot Cross-Country IO Detection — Macro-F1 Comparison',
             fontsize=16, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
# Separate the AVERAGE group visually.
ax.axvline(x=len(countries_display) - 0.5, color='#999999', linestyle=':', linewidth=1.2)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0',
          fontsize=10.5, framealpha=0.95)

# Value labels on the AVERAGE group only (keep per-country uncluttered).
for rects in (r1, r2, r3, r4):
    rect = rects[-1]
    ax.annotate(f'{rect.get_height():.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color='#111111', weight='bold')
# The story: every country selects λ*=1.0 except Venezuela, which selects λ*=0
# (alignment OFF) to avoid negative transfer. Call it out on the Venezuela group.
ven_i = countries.index("venezuela")
ax.annotate('λ*=0\n(alignment off →\navoids neg. transfer)',
            xy=(x[ven_i] + 1.5*width, adapt_m[ven_i]),
            xytext=(x[ven_i] + 1.5*width, 0.50),
            ha='center', va='top', fontsize=8.5, color='#4A0072', weight='bold',
            arrowprops=dict(arrowstyle='->', color='#4A0072', lw=1.3))
ax.annotate('(all others select λ*=1.0)', xy=(x[0], 0.05), ha='left', va='bottom',
            fontsize=8.5, color='#666666', style='italic')

plt.tight_layout()
out1 = os.path.join(HERE, "final_results_bar.png")
plt.savefig(out1, bbox_inches='tight')
print(f"Saved {out1}")

# ===========================================================================
# Chart 2: average-per-method headline bar
# ===========================================================================
def mean_of(vals):
    arr = [v for v in vals if not np.isnan(v)]
    return np.mean(arr)

methods = ['Paper\n(external)', 'no-DA\nrepro', 'DANN', 'CORAL', 'DFA-GFM\n(λ=1.0)', 'CAL-DFA-\nAdaptive']
avgs = [mean_of(paper_m), mean_of(repro_m), mean_of(dann_m), mean_of(coral_m),
        mean_of(dfa_m), mean_of(adapt_m)]
colors = [C_PAPER, C_REPRO, '#5AA9E6', '#2E86AB', C_DFA, C_OURS]
hatches = ['//', '', '', '', '', '']

fig2, ax2 = plt.subplots(figsize=(10, 6.5), dpi=300)
bars = ax2.bar(methods, avgs, color=colors, edgecolor='black', linewidth=0.9, width=0.62)
for b, h in zip(bars, hatches):
    if h:
        b.set_hatch(h)
# Reference line at the reproducible baseline.
ax2.axhline(y=mean_of(repro_m), color='#A0AAB2', linestyle='--', linewidth=1.2, alpha=0.8)
ax2.text(len(methods) - 0.5, mean_of(repro_m) + 0.004, 'reproducible baseline',
         ha='right', va='bottom', fontsize=9, color='#666666', style='italic')

ax2.set_ylabel('Average Test F1-Macro (6 countries)', fontsize=13, fontweight='bold', labelpad=10)
ax2.set_title('Average Zero-Shot Performance Across 6 Countries',
              fontsize=15, fontweight='bold', pad=16)
ax2.set_ylim(0.70, 0.81)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)
ax2.tick_params(axis='x', labelsize=10.5)
for b in bars:
    ax2.annotate(f'{b.get_height():.4f}',
                 xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, color='#111111', weight='bold')
# Highlight our winner.
bars[-1].set_linewidth(2.2)
plt.tight_layout()
out2 = os.path.join(HERE, "final_results_avg.png")
plt.savefig(out2, bbox_inches='tight')
print(f"Saved {out2}")
print(f"\nAdaptive selected lambdas: " + ", ".join(f"{c}={l:g}" for c, l in zip(countries, adapt_lams)))
print(f"Averages -> repro {mean_of(repro_m):.4f} | DFA {mean_of(dfa_m):.4f} | Ours {mean_of(adapt_m):.4f}")
