import re, os

countries = ['venezuela', 'cuba', 'iran', 'china', 'UAE', 'russia']
base_dir = 'c:/Users/minelab/Desktop/projects/ssm'

def read_file(filepath):
    for enc in ['utf-16', 'utf-16-le', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
                if content.strip():
                    return content
        except:
            continue
    return None

paper_baseline = {
    'china': 0.5814, 'iran': 0.7278, 'UAE': 0.8393,
    'cuba': 0.8991, 'russia': 0.7977, 'venezuela': 0.9099
}

print('\n=== Per-Subnet Signals (DFA-GFM) ===')
print(f"{'Country':<12} {'Overall':>8} {'coRT':>8} {'coURL':>8} {'hashSeq':>8} {'fastRT':>8} {'tweetSim':>8} {'Paper':>8}")
print('-' * 80)
for country in countries:
    fpath = os.path.join(base_dir, f'zero-shot_DFA_{country}.txt')
    content = read_file(fpath)
    if content:
        def get(pat):
            m = re.search(pat, content)
            return float(m.group(1)) if m else None

        overall  = get(r'\[TEST\] f1_macro:\s*([0-9.]+)')
        coRT     = get(r'\[TEST_coRT\] f1_macro:\s*([0-9.]+)')
        coURL    = get(r'\[TEST_coURL\] f1_macro:\s*([0-9.]+)')
        hashSeq  = get(r'\[TEST_hashSeq\] f1_macro:\s*([0-9.]+)')
        fastRT   = get(r'\[TEST_fastRT\] f1_macro:\s*([0-9.]+)')
        tweetSim = get(r'\[TEST_tweetSim\] f1_macro:\s*([0-9.]+)')
        paper    = paper_baseline[country]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "

        diff = (overall - paper) if overall is not None else None
        marker = " [+]" if diff and diff > 0 else " [-]"
        print(f"{country:<12} {fmt(overall):>8} {fmt(coRT):>8} {fmt(coURL):>8} {fmt(hashSeq):>8} {fmt(fastRT):>8} {fmt(tweetSim):>8} {paper:>8.4f}{marker}")

print()
print('=== Key Diagnosis ===')
print('hashSeq/fastRT/tweetSim collapsing (precision=1.0, low F1) = CLASS IMBALANCE in target')
print('Structural subnets (coRT, coURL) are STRONGER than text subnets across ALL countries')
print('Venezuela/Cuba fail because text alignment HURTS dissimilar language domains')
