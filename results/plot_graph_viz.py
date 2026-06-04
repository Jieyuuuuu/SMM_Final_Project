"""
Graph Visualization for IOHunter: Data → Graph Transformation
Shows how raw social media data is converted into 5 behavioral similarity networks.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── Load Russia data (smallest, most manageable for visualization) ──────────
with open('InfoOpsGFM/data/processed/russia/0.7_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

labels = np.array(data['labels'])  # 0=normal, 1=IO
IO_COLOR   = '#EF476F'   # Coral Red for IO accounts
NORM_COLOR = '#4CC9F0'   # Sky Blue for normal accounts
EDGE_COLOR = '#AAAAAA'

subnet_info = {
    'coRT':     {'color': '#06D6A0', 'title': 'co-Retweet (coRT)',    'desc': 'Users who retweeted\nthe same posts'},
    'coURL':    {'color': '#4361EE', 'title': 'co-URL (coURL)',        'desc': 'Users who shared\nthe same URLs'},
    'hashSeq':  {'color': '#F72585', 'title': 'co-Hashtag (hashSeq)', 'desc': 'Users who used\nthe same hashtags'},
    'fastRT':   {'color': '#FFAA00', 'title': 'Fast Retweet (fastRT)','desc': 'Users who retweeted\nwithin seconds'},
    'tweetSim': {'color': '#7209B7', 'title': 'Tweet Similarity',     'desc': 'Users with similar\ntweet content (SBERT)'},
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Data Pipeline Overview - "Raw Data → Graph"
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(16, 5), dpi=200)
ax.axis('off')
fig1.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

# Background
fig1.patch.set_facecolor('#0D1117')

steps = [
    {'label': 'Raw Twitter\nData',        'sub': '~716 Users\n~10K+ Tweets',      'color': '#1C2333', 'icon': '[Raw]'},
    {'label': 'User-Level\nAggregation',  'sub': 'Top-5 tweets\nper user (SBERT)', 'color': '#1C2333', 'icon': '[User]'},
    {'label': 'Signal\nExtraction',       'sub': '5 coordination\npattern types',   'color': '#1C2333', 'icon': '[Detect]'},
    {'label': 'Multi-Layer\nGraph Build', 'sub': 'Nodes=Users\nEdges=Similarity',  'color': '#1C2333', 'icon': '[Graph]'},
    {'label': 'GNN\nClassification',      'sub': 'IO vs Normal\naccount detection', 'color': '#1C2333', 'icon': '[GNN]'},
]
colors_step = ['#264653','#2a9d8f','#e9c46a','#f4a261','#e76f51']

x_positions = [0.10, 0.28, 0.46, 0.64, 0.82]
for i, (step, xpos, col) in enumerate(zip(steps, x_positions, colors_step)):
    # Box
    box = mpatches.FancyBboxPatch((xpos-0.08, 0.15), 0.15, 0.70,
                                   boxstyle="round,pad=0.02",
                                   facecolor=col, edgecolor='white',
                                   linewidth=1.5, alpha=0.9,
                                   transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(xpos, 0.72, step['icon'], ha='center', va='center',
            fontsize=22, transform=ax.transAxes)
    ax.text(xpos, 0.55, step['label'], ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(xpos, 0.36, step['sub'], ha='center', va='center',
            fontsize=8, color='#CCCCCC', transform=ax.transAxes)
    # Arrow
    if i < len(steps) - 1:
        ax.annotate('', xy=(x_positions[i+1]-0.085, 0.50),
                    xytext=(xpos+0.08, 0.50),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

ax.set_title('IOHunter: Social Media Data → Multi-Layer Behavioral Similarity Graph',
             fontsize=14, fontweight='bold', color='white', pad=15)

plt.tight_layout()
plt.savefig('graph_viz_pipeline.png', dpi=200, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved: graph_viz_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: 5 Subnet Networks Side-by-Side (Russia, subsampled)
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 4.5), dpi=200)
fig2.patch.set_facecolor('#0D1117')

for idx, (subnet_key, info) in enumerate(subnet_info.items()):
    ax = fig2.add_subplot(1, 5, idx + 1)
    ax.set_facecolor('#0D1117')

    G_full = data[subnet_key]
    # Subsample: take the largest connected component, max 120 nodes
    if G_full.number_of_nodes() > 120:
        largest_cc = max(nx.connected_components(G_full), key=len)
        G = G_full.subgraph(list(largest_cc)[:120]).copy()
    else:
        G = G_full.copy()

    node_list = list(G.nodes())
    node_labels_arr = labels[node_list]
    node_colors = [IO_COLOR if labels[n] == 1 else NORM_COLOR for n in node_list]
    node_sizes  = [60 if labels[n] == 1 else 30 for n in node_list]

    # Layout
    if G.number_of_nodes() < 5:
        pos = nx.spring_layout(G, seed=42)
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=0.5)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=info['color'],
                           alpha=0.25, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.9)
    ax.set_title(info['title'], fontsize=9, fontweight='bold',
                 color='white', pad=6)
    ax.text(0.5, -0.05, info['desc'], ha='center', va='top',
            fontsize=7, color='#AAAAAA', transform=ax.transAxes)
    ax.text(0.5, 1.0, f"N={G_full.number_of_nodes()}  E={G_full.number_of_edges()}",
            ha='center', va='bottom', fontsize=7.5, color=info['color'],
            transform=ax.transAxes, fontweight='bold')
    ax.axis('off')

# Legend
io_patch   = mpatches.Patch(color=IO_COLOR,   label='IO Account (malicious)')
norm_patch = mpatches.Patch(color=NORM_COLOR, label='Normal Account')
fig2.legend(handles=[io_patch, norm_patch], loc='lower center',
            ncol=2, fontsize=9, frameon=True, facecolor='#1C2333',
            edgecolor='#444', labelcolor='white', bbox_to_anchor=(0.5, -0.02))

fig2.suptitle('Russia: 5 Behavioral Similarity Sub-Networks (IO Accounts in Red)',
              fontsize=13, fontweight='bold', color='white', y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('graph_viz_subnets.png', dpi=200, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved: graph_viz_subnets.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Coordination Clique Zoom-In (coRT - most informative)
# Shows how IO accounts cluster together in the retweet network
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
fig3.patch.set_facecolor('#0D1117')

G_coRT = data['coRT']

# Left: full coRT network (subsampled)
ax = axes[0]
ax.set_facecolor('#0D1117')
largest_cc = max(nx.connected_components(G_coRT), key=len)
G_sub = G_coRT.subgraph(list(largest_cc)[:200]).copy()
node_list = list(G_sub.nodes())
node_colors = [IO_COLOR if labels[n] == 1 else NORM_COLOR for n in node_list]
node_sizes  = [80 if labels[n] == 1 else 25 for n in node_list]

pos = nx.spring_layout(G_sub, seed=42, k=0.4)
nx.draw_networkx_edges(G_sub, pos, ax=ax, edge_color='#06D6A0',
                       alpha=0.2, width=0.6)
nx.draw_networkx_nodes(G_sub, pos, ax=ax, nodelist=node_list,
                       node_color=node_colors, node_size=node_sizes, alpha=0.9)
ax.set_title('co-Retweet Network (Russia)\nIO accounts cluster in dense cliques',
             fontsize=10, fontweight='bold', color='white', pad=8)

io_count = sum(1 for n in node_list if labels[n] == 1)
norm_count = len(node_list) - io_count
ax.text(0.02, 0.02, f'IO={io_count}  Normal={norm_count}',
        transform=ax.transAxes, fontsize=8, color='white',
        bbox=dict(boxstyle='round', facecolor='#1C2333', alpha=0.8))
ax.axis('off')

# Right: IO-only subgraph (the "clique" IO accounts form)
ax = axes[1]
ax.set_facecolor('#0D1117')

io_nodes = [n for n in G_coRT.nodes() if labels[n] == 1]
G_io = G_coRT.subgraph(io_nodes).copy()
# Get densest subgraph (top degree nodes)
degrees = dict(G_io.degree())
top_io = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:60]
G_io_sub = G_io.subgraph(top_io).copy()

pos_io = nx.spring_layout(G_io_sub, seed=42, k=0.3)
nx.draw_networkx_edges(G_io_sub, pos_io, ax=ax, edge_color=IO_COLOR,
                       alpha=0.4, width=1.0)
nx.draw_networkx_nodes(G_io_sub, pos_io, ax=ax,
                       nodelist=list(G_io_sub.nodes()),
                       node_color=IO_COLOR, node_size=100, alpha=0.95)
ax.set_title('IO Account Sub-Graph (Coordination Clique)\nTightly connected = coordinated behavior',
             fontsize=10, fontweight='bold', color='white', pad=8)
ax.text(0.02, 0.02, 'Only IO accounts shown\nDense connections = synchronized posting',
        transform=ax.transAxes, fontsize=8, color='#FFB3C1',
        bbox=dict(boxstyle='round', facecolor='#1C2333', alpha=0.8))
ax.axis('off')

io_patch   = mpatches.Patch(color=IO_COLOR,   label='IO Account (malicious)')
norm_patch = mpatches.Patch(color=NORM_COLOR, label='Normal Account')
fig3.legend(handles=[io_patch, norm_patch], loc='lower center', ncol=2,
            fontsize=9, frameon=True, facecolor='#1C2333',
            edgecolor='#444', labelcolor='white', bbox_to_anchor=(0.5, -0.03))

fig3.suptitle('Key Insight: IO Accounts Form Dense Coordination Cliques in the co-Retweet Network',
              fontsize=12, fontweight='bold', color='white', y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('graph_viz_coordination_clique.png', dpi=200, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved: graph_viz_coordination_clique.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cross-Country Graph Statistics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
countries = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']
stats = {}
for country in countries:
    try:
        fpath = f'InfoOpsGFM/data/processed/{country}/0.7_datasets.pkl'
        with open(fpath, 'rb') as f:
            d = pickle.load(f)
        g = d['graph']
        lbl = np.array(d['labels'])
        stats[country] = {
            'nodes': g.number_of_nodes(),
            'edges': g.number_of_edges(),
            'io_ratio': float(lbl.sum()) / len(lbl) * 100,
            'density': nx.density(g) * 100,
        }
    except Exception as e:
        print(f"  Skip {country}: {e}")

fig4, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
fig4.patch.set_facecolor('#0D1117')
country_labels = [c.upper() for c in countries]
BAR_COLOR = '#4361EE'
ACCENT    = '#06D6A0'

# Plot 1: Node counts
ax = axes[0]
ax.set_facecolor('#0D1117')
vals = [stats[c]['nodes'] for c in countries]
bars = ax.bar(country_labels, vals, color=BAR_COLOR, edgecolor='white', linewidth=0.5, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, f'{v:,}',
            ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_title('Number of User Nodes', fontsize=11, color='white', fontweight='bold')
ax.set_ylabel('Count', color='#AAAAAA', fontsize=9)
ax.tick_params(colors='white', labelsize=8)
ax.set_facecolor('#0D1117')
for spine in ax.spines.values(): spine.set_color('#333')
ax.yaxis.label.set_color('#AAAAAA')

# Plot 2: IO account ratio
ax = axes[1]
ax.set_facecolor('#0D1117')
vals = [stats[c]['io_ratio'] for c in countries]
bars = ax.bar(country_labels, vals, color=IO_COLOR, edgecolor='white', linewidth=0.5, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
            ha='center', fontsize=8, color='white', fontweight='bold')
ax.axhline(y=50, color='white', linestyle='--', linewidth=0.8, alpha=0.4, label='50% line')
ax.set_title('IO Account Ratio (%)', fontsize=11, color='white', fontweight='bold')
ax.set_ylabel('IO Ratio (%)', color='#AAAAAA', fontsize=9)
ax.tick_params(colors='white', labelsize=8)
ax.set_facecolor('#0D1117')
for spine in ax.spines.values(): spine.set_color('#333')
ax.set_ylim(0, 70)

# Plot 3: Graph density
ax = axes[2]
ax.set_facecolor('#0D1117')
vals = [stats[c]['density'] for c in countries]
bars = ax.bar(country_labels, vals, color=ACCENT, edgecolor='white', linewidth=0.5, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{v:.2f}%',
            ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_title('Graph Density (%)', fontsize=11, color='white', fontweight='bold')
ax.set_ylabel('Density (%)', color='#AAAAAA', fontsize=9)
ax.tick_params(colors='white', labelsize=8)
ax.set_facecolor('#0D1117')
for spine in ax.spines.values(): spine.set_color('#333')

fig4.suptitle('Cross-Country Dataset Statistics: Graph Structure Comparison',
              fontsize=13, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('graph_viz_dataset_stats.png', dpi=200, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved: graph_viz_dataset_stats.png")

print("\nAll 4 graph visualization charts generated successfully!")
