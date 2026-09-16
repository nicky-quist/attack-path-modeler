"""
Does message passing earn its place? An ablation study.

benchmark.py reproduces the published table: a GCN beats logistic regression
by 0.143 F1 over five networks, and the gap was attributed to message passing.
That comparison changed two things at once: the GCN has message passing *and*
a nonlinear classifier head, and logistic regression has neither. This study
separates them, and asks three questions the original couldn't:

  1. Architecture-matched control. Against a model identical to the GCN except
     that no information crosses an edge (src/models.py, conv="none"), what is
     message passing worth?
  2. Feature ablation. The `criticality` feature is a per-zone constant, so it
     works as a zone label, and in this estate zone roughly determines distance
     to the data tier. Does message passing matter more once the model can't
     read position straight off a feature?
  3. Unseen networks. Every published number trains and tests on edges of the
     *same* graph. Does anything carry over to a network the model has never
     seen, which is what scoring a new estate would require?

Protocols
  within-network  the published protocol: per network, an 80/20 edge split,
                  messages pass along training edges only.
  unseen-network  5-fold over networks: train on 24, score each of the other
                  6. At scoring time messages pass along the unseen network's
                  edges, since topology is an observed input; its labels never
                  reach training (tests/test_models.py checks this).

Every model sees the same features on the same split, trained with the same
seed, optimiser and class weighting. Networks with no positive edge to find
are excluded and counted, since F1 and AP are undefined there.

    python experiments/study.py            # 30 networks, writes experiments/results/
    python experiments/study.py --quick    # 6 networks, 2 folds, smoke test
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from src.baselines import (cvss_threshold_baseline, logistic_regression_baseline,  # noqa: E402
                           majority_baseline)
from src.exploitability import annotate_hosts  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.gnn import build_pyg_data  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.models import predict_proba, train_edge_model  # noqa: E402
from src.stats import (average_precision, bootstrap_ci, brier, ece,  # noqa: E402
                       paired_summary)
from src.synthetic import generate_synthetic_network  # noqa: E402

FEATURE_SETS = {
    "all": [],
    "no_criticality": ["criticality"],
    # vulnerability evidence only: no zone proxy, no local structure
    "vuln_only": ["criticality", "in_degree", "out_degree"],
}
NEURAL = ["none", "gcn", "sage", "gat"]
LABEL = {"none": "no message passing", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT",
         "logreg": "logistic regression", "majority": "majority class",
         "cvss": "max_cvss > 8.5"}


def load_network(seed, train_frac):
    hosts, policy = generate_synthetic_network(seed=seed)
    G = build_graph(annotate_hosts(hosts, offline=True), policy)
    return G, build_pyg_data(G, policy, train_frac=train_frac, seed=seed)


def with_features_removed(data, names):
    """Zero the named columns. A constant column carries no information, and
    keeping the width fixed means every feature set trains the same architecture."""
    out = data.clone()
    if names:
        out.x = data.x.clone()
        for name in names:
            out.x[:, FEATURE_NAMES.index(name)] = 0.0
    return out


def logreg_proba(edge_x, y, train_mask, epochs=400, lr=0.05):
    """baselines.logistic_regression_baseline, returning probabilities instead of labels."""
    torch.manual_seed(0)
    model = torch.nn.Linear(edge_x.size(1), 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    counts = torch.bincount(y[train_mask], minlength=2).float()
    weight = counts.sum() / (2.0 * counts.clamp(min=1))
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(edge_x[train_mask]), y[train_mask],
                                          weight=weight).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(edge_x), dim=1)[:, 1]


def score(probs, labels):
    """F1 at the argmax boundary (as published), plus threshold-free and calibration metrics."""
    p, y = probs.tolist(), labels.tolist()
    pred = (probs >= 0.5).long()
    return {
        "f1": binary_metrics(pred, labels)["f1"],
        "ap": average_precision(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "n_edges": len(y),
        "n_pos": int(sum(y)),
    }


def within_network(seeds, log):
    records, excluded = [], []
    for seed in seeds:
        G, data = load_network(seed, train_frac=0.8)
        train_pos, test_pos = int(data.y[data.train_mask].sum()), int(data.y[data.test_mask].sum())
        if train_pos == 0 or test_pos == 0:
            excluded.append({"seed": seed, "train_pos": train_pos, "test_pos": test_pos})
            log(f"  network {seed:2d}: excluded ({train_pos} train / {test_pos} test positives)")
            continue

        test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()
        y_test = data.y[data.test_mask]
        base = {"protocol": "within", "seed": seed}

        # Label-only baselines don't use learned features, so they run once.
        for name, m in (("majority", majority_baseline(data.y[data.train_mask], y_test)),
                        ("cvss", cvss_threshold_baseline(G, data.edges, y_test, test_idx))):
            records.append({**base, "features": "all", "model": name, "f1": m["f1"],
                            "ap": None, "brier": None, "ece": None,
                            "n_edges": len(y_test), "n_pos": test_pos})

        for fs, drop in FEATURE_SETS.items():
            d = with_features_removed(data, drop)
            edge_x = torch.cat([d.x[d.edge_index[0]], d.x[d.edge_index[1]]], dim=1)

            probs = logreg_proba(edge_x, d.y, d.train_mask)
            rec = score(probs[d.test_mask], y_test)
            if fs == "all":
                # guard: the probability version must agree with the published baseline
                published = logistic_regression_baseline(edge_x, d.y, d.train_mask, d.test_mask)
                assert abs(rec["f1"] - published["f1"]) < 1e-9, (seed, rec["f1"], published["f1"])
            records.append({**base, "features": fs, "model": "logreg", **rec})

            for conv in NEURAL:
                model = train_edge_model(conv, [d])
                probs = predict_proba(model, d)
                records.append({**base, "features": fs, "model": conv,
                                **score(probs[d.test_mask], y_test)})
        log(f"  network {seed:2d}: done ({test_pos} test positives)")
    return records, excluded


def unseen_network(seeds, n_folds, feature_sets, log):
    records, excluded = [], []
    graphs = {}
    for seed in seeds:
        _, d = load_network(seed, train_frac=1.0)  # all edges trainable, all edges carry messages
        graphs[seed] = d

    for fold in range(n_folds):
        test_seeds = [s for i, s in enumerate(seeds) if i % n_folds == fold]
        train_seeds = [s for s in seeds if s not in test_seeds]
        for fs in feature_sets:
            drop = FEATURE_SETS[fs]
            train = [with_features_removed(graphs[s], drop) for s in train_seeds]
            for conv in NEURAL:
                model = train_edge_model(conv, train)
                for s in test_seeds:
                    test = with_features_removed(graphs[s], drop)
                    if int(test.y.sum()) == 0:
                        continue
                    records.append({"protocol": "unseen", "seed": s, "fold": fold, "features": fs,
                                    "model": conv, **score(predict_proba(model, test), test.y)})
        log(f"  fold {fold + 1}/{n_folds}: trained on {len(train_seeds)} networks, "
            f"scored {len(test_seeds)}")
    excluded = [{"seed": s, "total_pos": 0} for s in seeds if int(graphs[s].y.sum()) == 0]
    return records, excluded


def summarise(records, n_boot):
    groups = {}
    for r in records:
        groups.setdefault((r["protocol"], r["features"], r["model"]), []).append(r)

    models = []
    for (protocol, fs, model), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        entry = {"protocol": protocol, "features": fs, "model": model, "n": len(rows)}
        for metric in ("f1", "ap"):
            vals = [r[metric] for r in rows if r[metric] is not None]
            if vals:
                est, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
                entry[metric] = {"mean": est, "ci_low": lo, "ci_high": hi}
        for metric in ("brier", "ece"):
            vals = [r[metric] for r in rows if r[metric] is not None]
            if vals:
                entry[metric] = sum(vals) / len(vals)
        models.append(entry)

    def by_seed(protocol, fs, model, metric):
        return {r["seed"]: r[metric] for r in groups.get((protocol, fs, model), [])
                if r[metric] is not None}

    comparisons = []

    def compare(protocol, fs_a, model_a, fs_b, model_b, question):
        for metric in ("f1", "ap"):
            a, b = by_seed(protocol, fs_a, model_a, metric), by_seed(protocol, fs_b, model_b, metric)
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                continue
            comparisons.append({
                "protocol": protocol, "question": question, "metric": metric,
                "a": f"{LABEL[model_a]} [{fs_a}]", "b": f"{LABEL[model_b]} [{fs_b}]",
                **paired_summary([a[s] for s in common], [b[s] for s in common], n_boot=n_boot),
            })

    for protocol in ("within", "unseen"):
        fss = sorted({fs for (p, fs, _) in groups if p == protocol},
                     key=list(FEATURE_SETS).index)
        if protocol == "within":
            compare(protocol, "all", "gcn", "all", "logreg", "published claim: GCN vs logistic regression")
            compare(protocol, "all", "none", "all", "logreg", "nonlinear head alone (no message passing)")
        for fs in fss:
            for conv in ("gcn", "sage", "gat"):
                compare(protocol, fs, conv, fs, "none", "message passing, architecture held fixed")
        for model in ("none", "gcn"):
            if "vuln_only" in fss:
                compare(protocol, "all", model, "vuln_only", model, "cost of losing positional features")
    return models, comparisons


def fmt_ci(d):
    return f"{d['mean']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]" if d else "—"


def write_markdown(path, meta, models, comparisons):
    L = ["# Ablation study: does message passing earn its place?", "",
         f"Generated by `python experiments/study.py` on {meta['generated']}. "
         f"{meta['n_networks']} synthetic networks, {meta['n_boot']:,} bootstrap resamples. "
         "Means with 95% bootstrap intervals over networks. Paired comparisons use the "
         "per-network difference on identical splits; *p* is an exact two-sided sign test.", ""]
    for protocol, title in (("within", "Within-network (published protocol)"),
                            ("unseen", f"Unseen networks ({meta['folds']}-fold over networks)")):
        rows = [m for m in models if m["protocol"] == protocol]
        if not rows:
            continue
        excl = meta["excluded"][protocol]
        L += [f"## {title}", "",
              f"Networks scored: {max(m['n'] for m in rows)}. "
              f"Excluded, no positive edges to find: {len(excl)} "
              f"({', '.join(str(e['seed']) for e in excl) or 'none'}).", "",
              "| Features | Model | F1 | Average precision | Brier | ECE |",
              "|---|---|---|---|---|---|"]
        order = {k: i for i, k in enumerate(["majority", "cvss", "logreg", "none", "gcn", "sage", "gat"])}
        for m in sorted(rows, key=lambda m: (list(FEATURE_SETS).index(m["features"]), order[m["model"]])):
            brier_ = f"{m['brier']:.3f}" if "brier" in m else "—"
            ece_ = f"{m['ece']:.3f}" if "ece" in m else "—"
            L.append(f"| {m['features']} | {LABEL[m['model']]} | {fmt_ci(m.get('f1'))} | "
                     f"{fmt_ci(m.get('ap'))} | {brier_} | {ece_} |")
        L += ["", "| Question | Metric | A − B | Mean difference [95% CI] | A wins / ties / losses | *p* |",
              "|---|---|---|---|---|---|"]
        for c in (c for c in comparisons if c["protocol"] == protocol):
            L.append(f"| {c['question']} | {c['metric'].upper()} | {c['a']} − {c['b']} | "
                     f"{c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
                     f"{c['wins']} / {c['ties']} / {c['losses']} | {c['sign_test_p']:.3g} |")
        L.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--quick", action="store_true", help="6 networks, 2 folds, 1000 resamples")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()
    if args.quick:
        args.networks, args.folds, args.n_boot = 6, 2, 1000
        args.out = os.path.join(args.out, "quick")

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.networks))
    started = time.time()
    log = lambda msg: print(f"[{time.time() - started:6.0f}s] {msg}", flush=True)

    log(f"within-network protocol, {len(seeds)} networks")
    within, within_excl = within_network(seeds, log)
    log(f"unseen-network protocol, {args.folds} folds")
    unseen, unseen_excl = unseen_network(seeds, args.folds, ["all", "vuln_only"], log)

    records = within + unseen
    models, comparisons = summarise(records, args.n_boot)
    meta = {"generated": time.strftime("%Y-%m-%d"), "n_networks": len(seeds), "folds": args.folds,
            "n_boot": args.n_boot, "torch": torch.__version__, "seconds": round(time.time() - started),
            "excluded": {"within": within_excl, "unseen": unseen_excl}}

    with open(os.path.join(args.out, "study.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records, "models": models, "comparisons": comparisons}, f, indent=1)
    write_markdown(os.path.join(args.out, "STUDY.md"), meta, models, comparisons)
    log(f"wrote {args.out}/study.json and STUDY.md")


if __name__ == "__main__":
    main()
