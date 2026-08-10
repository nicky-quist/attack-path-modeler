import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # so this runs correctly regardless of cwd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which can't print em dashes/arrows

import matplotlib.pyplot as plt
import networkx as nx
from src.parser import parse_nessus
from src.graph import build_graph
from src.analysis import find_attack_paths, find_choke_points

hosts = parse_nessus("data/sample.nessus")
G = build_graph(hosts)

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

pos = nx.spring_layout(G, seed=42, k=2.5)

cvss_scores = [G.nodes[n]["max_cvss"] for n in G.nodes]
node_sizes = [G.nodes[n]["max_cvss"] * 400 for n in G.nodes]

nodes = nx.draw_networkx_nodes(G, pos,
                                node_color=cvss_scores,
                                cmap=plt.cm.RdYlGn_r,
                                vmin=0, vmax=10,
                                node_size=node_sizes,
                                ax=ax)

labels = {n: f"{G.nodes[n]['hostname']}\nCVSS: {G.nodes[n]['max_cvss']}" for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels,
                         font_color="white", font_size=8,
                         font_weight="bold", ax=ax)

# Draw all edges in gray
nx.draw_networkx_edges(G, pos,
                        edge_color="#444466",
                        arrows=True,
                        arrowsize=10,
                        width=1,
                        connectionstyle="arc3,rad=0.1",
                        ax=ax)

# Highlight the worst attack path in red
results = find_attack_paths(G, "10.0.0.1", "10.0.1.4")
worst_path = results["shortest"]
path_edges = list(zip(worst_path, worst_path[1:]))
nx.draw_networkx_edges(G, pos,
                        edgelist=path_edges,
                        edge_color="red",
                        arrows=True,
                        arrowsize=20,
                        width=3,
                        connectionstyle="arc3,rad=0.1",
                        ax=ax)

cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
cbar.set_label("Max CVSS Score", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

legend_labels = ["Critical (9-10)", "High (7-8.9)", "Medium (4-6.9)", "Highest Risk Path"]
legend_colors = ["#d73027", "#fc8d59", "#4dac26", "red"]
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=c, markersize=12, label=l)
           for c, l in zip(legend_colors, legend_labels)]
ax.legend(handles=handles, loc="upper left",
          facecolor="#2a2a4a", labelcolor="white", fontsize=9)

path_str = " → ".join([G.nodes[n]["hostname"] for n in worst_path])
ax.set_title(f"Attack Path Graph\nHighest Risk Path: {path_str}",
             color="white", fontsize=13, pad=20)
ax.axis("off")
plt.tight_layout()
plt.show()

from src.export import export_graph
export_graph(G, "data/graph.json", risk_path=worst_path, source_label="data/sample.nessus (fixed demo dataset)")
from src.gnn import build_pyg_data, train_gnn, generate_synthetic_data
print("\n--- GNN Training (synthetic data) ---")
synthetic_hosts = generate_synthetic_data(100)
from src.graph import build_graph as bg
G_synthetic = bg(synthetic_hosts)
pyg_data = build_pyg_data(synthetic_hosts, G_synthetic)
print(f"Synthetic graph: {len(synthetic_hosts)} nodes, {G_synthetic.number_of_edges()} edges")
model = train_gnn(pyg_data)