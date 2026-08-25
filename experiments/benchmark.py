"""
Repeated evaluation across independent networks and splits.

A single run of a single split is not a result — it is an anecdote, and the
previous version of this project reported one. Every number in the README comes
from this script: a fresh synthetic estate and a fresh train/test split per
seed, reported as mean +/- standard deviation.

    python experiments/benchmark.py [n_seeds]
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.exploitability import annotate_hosts
from src.gnn import build_pyg_data, evaluate_with_baselines
from src.graph import build_graph
from src.synthetic import generate_synthetic_network


def run(n_seeds=5):
    collected = {}
    shapes = []

    for seed in range(n_seeds):
        hosts, policy = generate_synthetic_network(seed=seed)
        hosts = annotate_hosts(hosts, offline=True)
        G = build_graph(hosts, policy)
        data = build_pyg_data(G, policy, seed=seed)
        shapes.append((G.number_of_nodes(), G.number_of_edges(),
                       int(data.y.sum()) / len(data.y)))

        print(f"--- seed {seed}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, "
              f"{int(data.y.sum()) / len(data.y) * 100:.1f}% positive ---")
        rows = evaluate_with_baselines(G, policy, data=data, verbose=False)
        for name, m in rows:
            collected.setdefault(name, []).append(m["f1"])
            print(f"    {name:<34} F1 {m['f1']:.3f}")
        print()

    print("=" * 66)
    print(f"F1 over {n_seeds} independent networks and splits (mean +/- sd)")
    print("=" * 66)
    for name, scores in collected.items():
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{name:<34} {statistics.mean(scores):.3f} +/- {sd:.3f}")

    avg_nodes = statistics.mean(s[0] for s in shapes)
    avg_edges = statistics.mean(s[1] for s in shapes)
    avg_pos = statistics.mean(s[2] for s in shapes)
    print(f"\ngraphs: {avg_nodes:.0f} nodes, {avg_edges:.0f} edges, "
          f"{avg_pos * 100:.1f}% positive (mean)")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
