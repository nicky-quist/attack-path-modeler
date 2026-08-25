"""
Regression tests for the two leaks that made the original results meaningless.

These are the tests that would have caught the problem in the first place, so
they are the ones that matter most: the first fails if the label ever again
becomes a function of an input feature, and the second fails if held-out edges
are ever again propagated through the network before being scored.
"""
import math
import unittest

import torch

from src.exploitability import annotate_hosts
from src.gnn import build_pyg_data
from src.graph import build_graph
from src.labels import label_edges
from src.synthetic import generate_synthetic_network


def _fixture(seed=0):
    hosts, policy = generate_synthetic_network(seed=seed)
    hosts = annotate_hosts(hosts, offline=True)
    return build_graph(hosts, policy), policy


class TestLabelIsNotLeaked(unittest.TestCase):
    """The original target was `target.max_cvss > 8.5` while max_cvss was also
    input feature 0, so a one-line threshold scored 100%. If that ever comes
    back, this fails."""

    def test_threshold_on_target_cvss_does_not_solve_the_task(self):
        """Under the original label this scored a perfect 1.000, because the
        label *was* this rule. It should no longer be able to.

        The bar is deliberately not "no correlation". An optimal attacker move
        does genuinely tend toward exploitable hosts, so the structural label
        correlates with target severity and should - an oracle-tuned threshold
        reaches F1 0.68-0.88 here, and the fixed 8.5 cut-point reported as a
        baseline averages 0.56. What must never return is *recovery*: a
        threshold reproducing the label outright, which is what a
        label-copies-feature bug looks like and what 1.000 meant.

        F1, not accuracy: at 13-26% positives, predicting all-negative already
        scores 0.74-0.87 accuracy while finding nothing at all.
        """
        from src.metrics import binary_metrics
        for seed in range(3):
            G, policy = _fixture(seed=seed)
            edges, labels = label_edges(G, policy)
            y = torch.tensor(labels)
            best = 0.0
            for threshold in [x / 2 for x in range(0, 21)]:
                pred = torch.tensor([1 if G.nodes[v]["max_cvss"] > threshold else 0
                                     for _, v in edges])
                best = max(best, binary_metrics(pred, y)["f1"])
            self.assertLess(best, 0.95,
                            f"seed {seed}: a CVSS threshold reproduced the label at "
                            f"F1 {best:.3f} - it has leaked into a node feature")

    def test_label_is_not_constant(self):
        G, policy = _fixture()
        _, labels = label_edges(G, policy)
        self.assertGreater(sum(labels), 0, "no positive labels")
        self.assertLess(sum(labels), len(labels), "no negative labels")

    def test_label_depends_on_graph_not_just_endpoints(self):
        """Identical endpoint features must be able to carry different labels;
        if they never do, the task is decidable without the topology."""
        G, policy = _fixture()
        edges, labels = label_edges(G, policy)
        by_target = {}
        for (_, v), lab in zip(edges, labels):
            by_target.setdefault(v, set()).add(lab)
        disagreeing = [v for v, labs in by_target.items() if len(labs) > 1]
        self.assertTrue(disagreeing,
                        "every edge into a given host shares a label — the target "
                        "node alone determines the answer")


class TestNoTransductiveLeakage(unittest.TestCase):
    """The original model ran message passing over every edge including the
    held-out ones, then evaluated on those same edges."""

    def test_message_passing_excludes_test_edges(self):
        G, policy = _fixture()
        data = build_pyg_data(G, policy, seed=0)
        mp = {(int(a), int(b)) for a, b in zip(*data.mp_edge_index.tolist())}
        test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()
        src, tgt = data.edge_index.tolist()
        for i in test_idx:
            self.assertNotIn((src[i], tgt[i]), mp,
                             "a held-out edge is being propagated through the graph")

    def test_masks_partition_the_edges(self):
        G, policy = _fixture()
        data = build_pyg_data(G, policy, seed=0)
        self.assertTrue(torch.all(data.train_mask ^ data.test_mask))
        self.assertEqual(int(data.train_mask.sum()) + int(data.test_mask.sum()),
                         len(data.y))

    def test_forbidden_features_absent(self):
        """crown-jewel status and distance-to-jewel must never be features."""
        from src.features import FEATURE_NAMES
        for banned in ("is_crown_jewel", "distance", "dist_to_jewel", "label"):
            self.assertNotIn(banned, FEATURE_NAMES)


class TestBellmanLabel(unittest.TestCase):
    def test_label_matches_the_optimality_condition(self):
        from src.labels import distance_to_crown_jewels
        G, policy = _fixture()
        dist = distance_to_crown_jewels(G, policy)
        edges, labels = label_edges(G, policy, tolerance=0.0)
        for (u, v), lab in zip(edges, labels):
            if u in dist and v in dist:
                advances = (G[u][v]["weight"] + dist[v]) <= dist[u] + 1e-9
                self.assertEqual(lab, 1 if advances else 0)

    def test_no_edge_beats_the_optimum(self):
        """d(u) must already be the best available move; nothing may undercut it."""
        from src.labels import distance_to_crown_jewels
        G, policy = _fixture()
        dist = distance_to_crown_jewels(G, policy)
        for u, v in G.edges():
            if u in dist and v in dist:
                self.assertGreaterEqual(G[u][v]["weight"] + dist[v], dist[u] - 1e-9)


if __name__ == "__main__":
    unittest.main()
