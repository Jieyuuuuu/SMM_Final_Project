import os
import re
import numpy as np

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
# Result .txt files live alongside this script, in the results/ directory.
base_dir = os.path.dirname(os.path.abspath(__file__))

def read_file(filepath):
    # Try different encodings due to PowerShell Tee-Object default to UTF-16LE
    for enc in ['utf-16', 'utf-16-le', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            continue
    return None

# NOTE: this is the legacy fixed-method comparison. The canonical final table,
# including the proposed CAL-DFA-Adaptive method, is produced by summarize_final.py.
results = {
    'baseline': {},
    'dann': {},
    'coral': {},
    'dfa': {},
    'repro': {}  # no-DA reproducible baseline (CAL-DFA script at coral_weight=0)
}

# 論文官方數據 (Table 3 Only PreTrain F1-Macro)
paper_baseline_f1 = {
    'china': '0.5814+-0.0589',
    'iran': '0.7278+-0.0143',
    'UAE': '0.8393+-0.0593',
    'cuba': '0.8991+-0.0535',
    'russia': '0.7977+-0.0193',
    'venezuela': '0.9099+-0.0107'
}

for country in countries:
    # Baseline
    b_file = os.path.join(base_dir, f"zero-shot_baseline_{country}.txt")
    if os.path.exists(b_file):
        content = read_file(b_file)
        if content:
            results['baseline'][country] = content

    # DANN
    d_file = os.path.join(base_dir, f"zero-shot_DANN_{country}.txt")
    if os.path.exists(d_file):
        content = read_file(d_file)
        if content:
            results['dann'][country] = content

    # CORAL
    c_file = os.path.join(base_dir, f"zero-shot_CORAL_{country}.txt")
    if os.path.exists(c_file):
        content = read_file(c_file)
        if content:
            results['coral'][country] = content

    # DFA
    dfa_file = os.path.join(base_dir, f"zero-shot_DFA_{country}.txt")
    if os.path.exists(dfa_file):
        content = read_file(dfa_file)
        if content:
            results['dfa'][country] = content

    # no-DA reproducible baseline (coral_weight=0)
    repro_file = os.path.join(base_dir, f"zero-shot_CW000_{country}.txt")
    if os.path.exists(repro_file):
        content = read_file(repro_file)
        if content:
            results['repro'][country] = content

# Parser metrics
def parse_metrics(content):
    metrics = {}
    patterns = {
        'acc': r'\[TEST\] accuracy:\s*([0-9\.\+-]+)',
        'prec': r'\[TEST\] precision:\s*([0-9\.\+-]+)',
        'f1_macro': r'\[TEST\] f1_macro:\s*([0-9\.\+-]+)',
        'f1_micro': r'\[TEST\] f1_micro:\s*([0-9\.\+-]+)',
    }
    
    for name, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            metrics[name] = m.group(1)
        else:
            metrics[name] = "N/A"
    return metrics

def f1_point(f1_str):
    """Return the mean F1 (float) from a '0.77+-0.03' string, or None."""
    try:
        if f1_str and f1_str != "N/A":
            return float(f1_str.split('+-')[0])
    except Exception:
        pass
    return None

print("\n" + "="*120)
print(f"{'Country':<12} | {'Paper (ext.)':<16} | {'no-DA repro':<16} | {'DANN':<16} | {'CORAL':<16} | {'DFA-GFM':<16}")
print("="*120)

# Accumulators for the average row.
sums = {'paper': [], 'repro': [], 'dann': [], 'coral': [], 'dfa': []}

for country in countries:
    b_f1 = paper_baseline_f1[country]
    r_f1 = "N/A"
    d_f1 = "N/A"
    c_f1 = "N/A"
    dfa_f1 = "N/A"

    if country in results['baseline']:
        bm = parse_metrics(results['baseline'][country])
        if bm.get('f1_macro') != "N/A":
            b_f1 = bm.get('f1_macro')

    if country in results['repro']:
        r_f1 = parse_metrics(results['repro'][country]).get('f1_macro')

    if country in results['dann']:
        d_f1 = parse_metrics(results['dann'][country]).get('f1_macro')

    if country in results['coral']:
        c_f1 = parse_metrics(results['coral'][country]).get('f1_macro')

    if country in results['dfa']:
        dfa_f1 = parse_metrics(results['dfa'][country]).get('f1_macro')

    # Track per-method means for averaging.
    for key, val in (('paper', b_f1), ('repro', r_f1), ('dann', d_f1),
                     ('coral', c_f1), ('dfa', dfa_f1)):
        pt = f1_point(val)
        if pt is not None:
            sums[key].append(pt)

    print(f"{country:<12} | {b_f1:<16} | {r_f1:<16} | {d_f1:<16} | {c_f1:<16} | {dfa_f1:<16}")

print("-"*120)
def avg(key):
    return f"{np.mean(sums[key]):.4f}" if sums[key] else "N/A"
print(f"{'AVERAGE':<12} | {avg('paper'):<16} | {avg('repro'):<16} | {avg('dann'):<16} | {avg('coral'):<16} | {avg('dfa'):<16}")
print("="*120)
print("\nNote: Paper numbers are external (Table 3) and NOT reproducible here; the")
print("'no-DA repro' column is the faithful in-codebase baseline (coral_weight=0).")
print("For the proposed CAL-DFA-Adaptive method and the full table, run summarize_final.py.")
