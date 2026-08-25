"""
Pipeline runner — Nessus scan input.

    python main.py                 # live EPSS + CISA KEV lookups
    python main.py --offline       # no network; CVSS-derived estimates
    python main.py --no-plot       # skip the matplotlib window
    python main.py --skip-gnn      # skip model training
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # runs correctly regardless of cwd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 can't print arrows/em dashes

import networkx as nx

from src.analysis import find_choke_points, most_probable_path, top_attack_paths
from src.exploitability import annotate_hosts
from src.export import export_graph
from src.graph import build_graph
from src.parser import parse_nessus
from src.segmentation import resolve_policy

OFFLINE = "--offline" in sys.argv
NO_PLOT = "--no-plot" in sys.argv
SKIP_GNN = "--skip-gnn" in sys.argv


def main():
    hosts = parse_nessus("data/sample.nessus")
    print(f"Parsed {len(hosts)} hosts from data/sample.nessus")

    print("Resolving exploitability" + (" (offline — CVSS estimates)" if OFFLINE
                                        else " (EPSS + CISA KEV)") + "...")
    hosts = annotate_hosts(hosts, offline=OFFLINE)

    policy = resolve_policy(hosts, "data/segmentation.json")
    print("\nSegmentation policy:")
    print("  " + policy.summary().replace("\n", "\n  "))

    G = build_graph(hosts, policy)
    n, e = G.number_of_nodes(), G.number_of_edges()
    print(f"\nAttack graph: {n} nodes, {e} edges "
          f"(density {e / (n * (n - 1)):.3f})")

    best = most_probable_path(G, policy)
    if best:
        chain = " -> ".join(G.nodes[x]["hostname"] for x in best["path"])
        print("\nMost probable attack chain:")
        print(f"  {chain}")
        print(f"  P(success) = {best['probability']:.4f}   "
              f"cost = {best['cost']}   hops = {best['hops']}")
        for a, b in zip(best["path"], best["path"][1:]):
            d = G[a][b]
            kev = "  [KEV]" if d.get("in_kev") else ""
            print(f"    {G.nodes[a]['hostname']:<18} -> {G.nodes[b]['hostname']:<18} "
                  f"{d['cve']} via {d['service']}/{d['port']}  "
                  f"p={d['p_exploit']:.4f} ({d['p_source']}){kev}")

    chokes = find_choke_points(G, policy, top=5)
    if chokes:
        print("\nChoke points — share of the top attack chains passing through:")
        for node, share, count, total in chokes:
            print(f"  {G.nodes[node]['hostname']:<20} {share:6.1%}  ({count}/{total} chains)")

    export_graph(
        G, "data/graph.json",
        risk_path=best["path"] if best else [],
        source_label="data/sample.nessus (fixed demo dataset)",
        path_probability=best["probability"] if best else None,
        choke_points=[{"id": c[0], "share": round(c[1], 4)} for c in chokes],
    )

    if not NO_PLOT:
        plot(G, best)

    if not SKIP_GNN:
        run_gnn()


def plot(G, best):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Colour by probability of exploitation rather than CVSS — the whole point
    # of the EPSS/KEV work is that these rank hosts differently.
    scores = [G.nodes[x].get("max_p_exploit", 0.0) for x in G.nodes]
    sizes = [400 + G.nodes[x].get("max_p_exploit", 0.0) * 2600 for x in G.nodes]

    nodes = nx.draw_networkx_nodes(G, pos, node_color=scores, cmap=plt.cm.RdYlGn_r,
                                   vmin=0, vmax=1, node_size=sizes, ax=ax)
    labels = {x: f"{G.nodes[x]['hostname']}\np={G.nodes[x].get('max_p_exploit', 0):.2f}"
              for x in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white",
                            font_size=8, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#444466", arrows=True, arrowsize=10,
                           width=1, connectionstyle="arc3,rad=0.1", ax=ax)

    if best:
        path_edges = list(zip(best["path"], best["path"][1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red",
                               arrows=True, arrowsize=20, width=3,
                               connectionstyle="arc3,rad=0.1", ax=ax)

    cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
    cbar.set_label("P(exploit) — EPSS / KEV", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    title = "Attack Path Graph"
    if best:
        chain = " → ".join(G.nodes[x]["hostname"] for x in best["path"])
        title += f"\nMost probable chain: {chain}  (P = {best['probability']:.3f})"
    ax.set_title(title, color="white", fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def run_gnn():
    from src.exploitability import annotate_hosts as annotate
    from src.gnn import evaluate_with_baselines
    from src.synthetic import generate_synthetic_network

    print("\n" + "=" * 66)
    print("Edge-risk model — synthetic segmented estate")
    print("=" * 66)
    print("Target: does this edge lie on an optimal route to a crown jewel?")
    print("Reported against every baseline on the same split. For mean +/- sd")
    print("across independent networks, run: python experiments/benchmark.py\n")

    hosts, policy = generate_synthetic_network()
    hosts = annotate(hosts, offline=True)  # fabricated CVEs — EPSS has no scores
    G = build_graph(hosts, policy)
    evaluate_with_baselines(G, policy)


if __name__ == "__main__":
    main()
