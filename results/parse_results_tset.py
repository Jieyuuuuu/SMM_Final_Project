import os
import re

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
base_dir = "c:/Users/minelab/Desktop/projects/ssm"

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

results = {
    'baseline': {},
    'dann': {},
    'coral': {},
    'dfa': {},
    'tset': {}
}

# Paper official numbers (Table 3, Only PreTrain F1-Macro)
paper_baseline_f1 = {
    'china': '0.5814+-0.0589',
    'iran': '0.7278+-0.0143',
    'UAE': '0.8393+-0.0593',
    'cuba': '0.8991+-0.0535',
    'russia': '0.7977+-0.0193',
    'venezuela': '0.9099+-0.0107'
}

for country in countries:
    for method in ['baseline', 'DANN', 'CORAL', 'DFA', 'TSET']:
        key = method.lower()
        fpath = os.path.join(base_dir, f"zero-shot_{method}_{country}.txt")
        if os.path.exists(fpath):
            content = read_file(fpath)
            if content:
                results[key][country] = content

def parse_metrics(content):
    metrics = {}
    patterns = {
        'acc':      r'\[TEST\] accuracy:\s*([0-9\.+-]+)',
        'prec':     r'\[TEST\] precision:\s*([0-9\.+-]+)',
        'f1_macro': r'\[TEST\] f1_macro:\s*([0-9\.+-]+)',
        'f1_micro': r'\[TEST\] f1_micro:\s*([0-9\.+-]+)',
    }
    for name, pat in patterns.items():
        m = re.search(pat, content)
        metrics[name] = m.group(1) if m else "N/A"
    return metrics

def to_float(s):
    try:
        return float(s.split('+-')[0])
    except Exception:
        return None

print("\n" + "=" * 130)
print(f"{'Country':<12} | {'Paper Baseline':^18} | {'DANN':^18} | {'CORAL':^18} | {'DFA-GFM':^18} | {'TSET-GFM':^18} | {'TSET vs Base':^12}")
print("=" * 130)

for country in countries:
    b_f1 = paper_baseline_f1[country]
    d_f1 = c_f1 = dfa_f1 = tset_f1 = "N/A"

    if country in results['baseline']:
        bm = parse_metrics(results['baseline'][country])
        if bm.get('f1_macro') != "N/A":
            b_f1 = bm.get('f1_macro')
    if country in results['dann']:
        d_f1 = parse_metrics(results['dann'][country]).get('f1_macro', 'N/A')
    if country in results['coral']:
        c_f1 = parse_metrics(results['coral'][country]).get('f1_macro', 'N/A')
    if country in results['dfa']:
        dfa_f1 = parse_metrics(results['dfa'][country]).get('f1_macro', 'N/A')
    if country in results['tset']:
        tset_f1 = parse_metrics(results['tset'][country]).get('f1_macro', 'N/A')

    # Compute delta TSET vs Paper Baseline
    b_val = to_float(b_f1)
    t_val = to_float(tset_f1)
    if b_val is not None and t_val is not None:
        delta = t_val - b_val
        flag = "[+]" if delta > 0 else "[-]"
        imp_str = f"{delta*100:+.2f}% {flag}"
    else:
        imp_str = "pending..."

    print(f"{country:<12} | {b_f1:<18} | {d_f1:<18} | {c_f1:<18} | {dfa_f1:<18} | {tset_f1:<18} | {imp_str:<12}")

print("=" * 130)

print("\n=== Detailed TSET-GFM Metrics ===")
for country in countries:
    if country in results['tset']:
        dm = parse_metrics(results['tset'][country])
        print(f"{country:<12} -> Acc: {dm.get('acc')}, Prec: {dm.get('prec')}, "
              f"F1-Macro: {dm.get('f1_macro')}, F1-Micro: {dm.get('f1_micro')}")
    else:
        print(f"{country:<12} -> [not yet computed]")

print("\n=== Per-Subnet TSET-GFM Analysis ===")
for country in countries:
    if country in results['tset']:
        content = results['tset'][country]
        def get_subnet_f1(subnet):
            m = re.search(rf'\[TEST_{subnet}\] f1_macro:\s*([0-9.]+)', content)
            return float(m.group(1)) if m else None
        overall = to_float(parse_metrics(content).get('f1_macro', 'N/A'))
        coRT     = get_subnet_f1('coRT')
        coURL    = get_subnet_f1('coURL')
        hashSeq  = get_subnet_f1('hashSeq')
        fastRT   = get_subnet_f1('fastRT')
        tweetSim = get_subnet_f1('tweetSim')
        paper = to_float(paper_baseline_f1[country])
        delta = f"{(overall - paper)*100:+.2f}%" if overall and paper else "N/A"
        print(f"{country:<12}  overall={overall:.4f}({delta})  "
              f"coRT={coRT:.4f}  coURL={coURL:.4f}  "
              f"hashSeq={hashSeq:.4f}  fastRT={fastRT:.4f}  tweetSim={tweetSim:.4f}"
              if all(v is not None for v in [overall, coRT, coURL, hashSeq, fastRT, tweetSim])
              else f"{country:<12}  [subnet data incomplete]")
    else:
        print(f"{country:<12}  [not yet computed]")
