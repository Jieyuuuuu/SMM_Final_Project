import os
import re

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
base_dir = "c:/Users/minelab/Desktop/projects/ssm"

# Official AAAI 2025 IOHunter Baseline values
paper_baseline_f1 = {
    'china': 0.5814,
    'iran': 0.7278,
    'UAE': 0.8393,
    'cuba': 0.8991,
    'russia': 0.7977,
    'venezuela': 0.9099
}

# Pre-Trained DFA-GFM values
dfa_f1 = {
    'china': 0.6026,
    'iran': 0.7063,
    'UAE': 0.8797,
    'cuba': 0.8559,
    'russia': 0.8362,
    'venezuela': 0.7676
}

def read_file(filepath):
    for enc in ['utf-16', 'utf-16-le', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            continue
    return None

print("="*120)
print(f"{'Country':<12} | {'Paper Baseline F1':<18} | {'DFA-GFM F1':<18} | {'Multi-Level GFM F1':<22} | {'Net Gain vs Base':<18} | {'Net Gain vs DFA':<18}")
print("="*120)

for country in countries:
    filename = f"zero-shot_multilevel_{country}.txt"
    filepath = os.path.join(base_dir, filename)
    
    ml_f1_str = "N/A"
    ml_f1_val = None
    
    if os.path.exists(filepath):
        content = read_file(filepath)
        if content:
            m = re.search(r'\[TEST\] f1_macro:\s*([0-9\.\+-]+)', content)
            if m:
                ml_f1_str = m.group(1)
                try:
                    ml_f1_val = float(ml_f1_str.split('+-')[0])
                except:
                    pass
                    
    base_val = paper_baseline_f1[country]
    dfa_val = dfa_f1[country]
    
    gain_vs_base = "N/A"
    gain_vs_dfa = "N/A"
    
    if ml_f1_val is not None:
        diff_base = ml_f1_val - base_val
        gain_vs_base = f"{diff_base*100:+.2f}%"
        
        diff_dfa = ml_f1_val - dfa_val
        gain_vs_dfa = f"{diff_dfa*100:+.2f}%"
        
    print(f"{country:<12} | {base_val*100:<17.2f}% | {dfa_val*100:<17.2f}% | {ml_f1_str:<22} | {gain_vs_base:<18} | {gain_vs_dfa:<18}")

print("="*120)
