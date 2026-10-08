"""
Repeated evaluation across independent networks and splits.

A single run of a single split is not a result — it is an anecdote, and the
previous version of this project reported one. Every number in the README comes
from this script: a fresh synthetic estate and a fresh train/test split per
seed, reported as mean +/- standard deviation.

Each network's GCN number is itself averaged over several training seeds (the
second argument, default 5), so dropout and initialisation noise — which alone
moves a single network by up to 0.17 F1 — is averaged out before the spread
across networks is reported. The baselines are deterministic given the split,
so they are run once.

    python experiments/benchmark.py [n_networks] [n_train_seeds]
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


def run(n_networks=5, n_train_seeds=5):
    # Reproducibility: per-training torch.manual_seed fixes dropout and init, but
    # multithreaded scatter in message passing reorders float adds, which drifts
    # the mean by ~0.004 between runs. Pin it so `python experiments/benchmark.py`
    # gives the same number every time.
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)

    collected = {}
    shapes = []
    gnn_seeds = range(n_train_seeds)

    for seed in range(n_networks):
        hosts, policy = generate_synthetic_network(seed=seed)
        hosts = annotate_hosts(hosts, offline=True)
        G = build_graph(hosts, policy)
        data = build_pyg_data(G, policy, seed=seed)
        shapes.append((G.number_of_nodes(), G.number_of_edges(),
                       int(data.y.sum()) / len(data.y)))

        print(f"--- network {seed}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, "
              f"{int(data.y.sum()) / len(data.y) * 100:.1f}% positive ---")
        rows = evaluate_with_baselines(G, policy, data=data, gnn_seeds=gnn_seeds, verbose=False)
        for name, m in rows:
            collected.setdefault(name, []).append(m["f1"])
            note = f"  (over {n_train_seeds} train seeds, sd {m['f1_sd']:.3f})" if "f1_sd" in m else ""
            print(f"    {name:<34} F1 {m['f1']:.3f}{note}")
        print()

    print("=" * 66)
    print(f"F1 over {n_networks} independent networks and splits (mean +/- sd)")
    print(f"GCN averaged over {n_train_seeds} training seeds per network")
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
    n_networks = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_train_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(n_networks, n_train_seeds)
