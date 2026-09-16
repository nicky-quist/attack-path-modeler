"""Tests for src/models.py.

The equivalence test is the one that matters. The ablation study attributes
differences to message passing, which is only valid if its GCN *is* the
published EdgeRiskGNN and its control differs from it in nothing but message
passing.
"""
import unittest

import torch

from src.exploitability import annotate_hosts
from src.gnn import build_pyg_data, train_gnn
from src.graph import build_graph
from src.metrics import binary_metrics
from src.models import EdgeClassifier, merge_graphs, predict_proba, train_edge_model
from src.synthetic import generate_synthetic_network


def graph_data(seed, train_frac=0.8):
    hosts, policy = generate_synthetic_network(seed=seed)
    G = build_graph(annotate_hosts(hosts, offline=True), policy)
    return build_pyg_data(G, policy, train_frac=train_frac, seed=seed)


class EquivalenceWithPublishedModel(unittest.TestCase):
    def test_gcn_reproduces_edge_risk_gnn_exactly(self):
        data = graph_data(seed=0)
        _, published = train_gnn(data, epochs=120, verbose=False)

        model = train_edge_model("gcn", [data], epochs=120)
        with torch.no_grad():
            pred = model(data.x, data.mp_edge_index, data.edge_index).argmax(dim=1)
        ours = binary_metrics(pred[data.test_mask], data.y[data.test_mask])
        self.assertEqual(ours, published, "study GCN must be the published model, not a lookalike")

    def test_control_matches_capacity_of_gcn(self):
        gcn = EdgeClassifier(in_dim=8, conv="gcn")
        none = EdgeClassifier(in_dim=8, conv="none")
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(gcn), count(none),
                         "the control must remove message passing, not capacity")


class NoMessagePassingControl(unittest.TestCase):
    def test_output_ignores_graph_structure(self):
        data = graph_data(seed=1)
        torch.manual_seed(0)
        model = EdgeClassifier(in_dim=data.x.size(1), conv="none").eval()
        with torch.no_grad():
            with_edges = model(data.x, data.mp_edge_index, data.edge_index)
            no_edges = model(data.x, torch.empty(2, 0, dtype=torch.long), data.edge_index)
        self.assertTrue(torch.equal(with_edges, no_edges))

    def test_gcn_output_does_depend_on_structure(self):
        data = graph_data(seed=1)
        torch.manual_seed(0)
        model = EdgeClassifier(in_dim=data.x.size(1), conv="gcn").eval()
        with torch.no_grad():
            with_edges = model(data.x, data.mp_edge_index, data.edge_index)
            no_edges = model(data.x, torch.empty(2, 0, dtype=torch.long), data.edge_index)
        self.assertFalse(torch.equal(with_edges, no_edges))


class MultiGraph(unittest.TestCase):
    def test_merge_offsets_indices_so_graphs_stay_disconnected(self):
        a, b = graph_data(seed=2), graph_data(seed=3)
        x, y, edge_index, mp, mask = merge_graphs([a, b])
        n_a = a.x.size(0)
        self.assertEqual(x.size(0), n_a + b.x.size(0))
        self.assertEqual(y.size(0), a.y.size(0) + b.y.size(0))
        a_edges, b_edges = edge_index[:, :a.edge_index.size(1)], edge_index[:, a.edge_index.size(1):]
        self.assertTrue(bool((a_edges < n_a).all()))
        self.assertTrue(bool((b_edges >= n_a).all()))
        self.assertTrue(bool((mp[:, a.mp_edge_index.size(1):] >= n_a).all()))

    def test_single_graph_passes_through_unchanged(self):
        a = graph_data(seed=2)
        _, _, edge_index, mp, _ = merge_graphs([a])
        self.assertTrue(torch.equal(edge_index, a.edge_index))
        self.assertTrue(torch.equal(mp, a.mp_edge_index))

    def test_predict_proba_is_a_probability_per_edge(self):
        data = graph_data(seed=4)
        for conv in ("none", "gcn", "sage", "gat"):
            model = train_edge_model(conv, [data], epochs=5)
            p = predict_proba(model, data)
            self.assertEqual(p.shape, (data.edge_index.size(1),), conv)
            self.assertTrue(bool(((p >= 0) & (p <= 1)).all()), conv)


class UnseenNetworkProtocol(unittest.TestCase):
    def test_test_network_labels_never_reach_training(self):
        """Training must not change if the unseen network's labels are scrambled."""
        train = [graph_data(seed=5, train_frac=1.0)]
        test = graph_data(seed=6, train_frac=1.0)

        m1 = train_edge_model("gcn", train, epochs=30)
        test.y = 1 - test.y  # scramble labels of the network we will score
        m2 = train_edge_model("gcn", train, epochs=30)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_unknown_conv_is_rejected(self):
        with self.assertRaises(ValueError):
            EdgeClassifier(in_dim=8, conv="transformer")


if __name__ == "__main__":
    unittest.main()
