"""
Edge-level risk classifier.

What this model is asked to predict — see labels.py — is whether an edge lies
on an optimal route from wherever it starts to a crown-jewel asset. That is a
global property of the graph. A node cannot answer it about itself, which is
the point: it is a task where message passing should earn its place, and it is
verifiable that it does, because the same features are handed to a linear model
that has no message passing at all.

Two structural corrections from the original:

*Leakage.* The old target was `target.max_cvss > 8.5` while `max_cvss` was
simultaneously input feature 0, fed to the classifier raw through the skip
connection. A single threshold scored 100%. That baseline is still run, on
purpose, so its collapse against the new target is visible.

*Transductive leakage.* The old model ran message passing over every edge in
the graph, including the held-out ones, then evaluated on those same edges.
Embeddings were therefore built partly from test data. Message passing now runs
over the training edges only; test edges are classified but never propagated.
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.features import node_feature_matrix
from src.labels import label_edges
from src.metrics import binary_metrics, format_table


def build_pyg_data(G, policy, train_frac=0.8, seed=42, tolerance=0.1):
    node_list = list(G.nodes())
    idx = {ip: i for i, ip in enumerate(node_list)}

    x = node_feature_matrix(G, node_list)
    edges, labels = label_edges(G, policy, tolerance=tolerance)

    edge_index = torch.tensor([[idx[u] for u, _ in edges],
                               [idx[v] for _, v in edges]], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(edges), generator=g)
    split = int(len(edges) * train_frac)
    train_mask = torch.zeros(len(edges), dtype=torch.bool)
    train_mask[perm[:split]] = True
    test_mask = ~train_mask

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)
    # Message passing sees training edges only — held-out edges are classified,
    # never propagated through.
    data.mp_edge_index = edge_index[:, train_mask]
    data.edges = edges
    data.node_list = node_list
    return data


class EdgeRiskGNN(torch.nn.Module):
    """Three GCN layers, because the estate's attack depth is three to four
    hops (internet -> dmz -> servers -> data) and a node needs to aggregate
    across that span to sense how far it sits from anything valuable.

    Raw features are concatenated back before classification so the head keeps
    an unsmoothed view of each endpoint alongside the propagated signal.
    """

    def __init__(self, in_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((hidden + in_dim) * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 2),
        )
        self.dropout = dropout

    def forward(self, x, mp_edge_index, target_edge_index):
        h = F.relu(self.conv1(x, mp_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, mp_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, mp_edge_index))
        h = torch.cat([h, x], dim=1)
        src, tgt = target_edge_index[0], target_edge_index[1]
        return self.classifier(torch.cat([h[src], h[tgt]], dim=1))


def train_gnn(data, epochs=400, lr=0.01, verbose=True):
    torch.manual_seed(42)
    model = EdgeRiskGNN(in_dim=data.x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
    weight = counts.sum() / (2.0 * counts.clamp(min=1))

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.mp_edge_index, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        opt.step()

        if verbose and epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.mp_edge_index, data.edge_index).argmax(dim=1)
            m = binary_metrics(pred[data.test_mask], data.y[data.test_mask])
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"test F1 {m['f1']:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.mp_edge_index, data.edge_index).argmax(dim=1)
    return model, binary_metrics(pred[data.test_mask], data.y[data.test_mask])


def evaluate_with_baselines(G, policy, data=None, verbose=True):
    """Train the GNN and report it beside every baseline on the same split."""
    from src.baselines import (cvss_threshold_baseline, logistic_regression_baseline,
                               majority_baseline)

    if data is None:
        data = build_pyg_data(G, policy)

    pos = int(data.y.sum())
    if verbose:
        print(f"edges: {len(data.y)}   positive: {pos} ({pos / len(data.y) * 100:.1f}%)")
        print(f"train/test: {int(data.train_mask.sum())}/{int(data.test_mask.sum())}\n")

    src_x = data.x[data.edge_index[0]]
    tgt_x = data.x[data.edge_index[1]]
    edge_x = torch.cat([src_x, tgt_x], dim=1)
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    y_test = data.y[data.test_mask]

    rows = [
        ("majority class", majority_baseline(data.y[data.train_mask], y_test)),
        ("cvss-threshold (ORIGINAL label)",
         cvss_threshold_baseline(G, data.edges, y_test, test_idx)),
        ("logistic regression (no graph)",
         logistic_regression_baseline(edge_x, data.y, data.train_mask, data.test_mask)),
    ]

    if verbose:
        print("training GNN...")
    _, gnn_metrics = train_gnn(data, verbose=verbose)
    rows.append(("GCN (3-layer, message passing)", gnn_metrics))

    if verbose:
        print()
        print(format_table(rows))
    return rows
