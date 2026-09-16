"""
Edge classifiers for the ablation study, with message passing as the one variable.

`EdgeRiskGNN` in gnn.py was compared against logistic regression, and the gap
was credited to message passing. But the GCN also has a nonlinear classifier
head that logistic regression lacks, so that comparison changed two things at
once. `EdgeClassifier` holds everything fixed (depth, width, dropout, the
skip connection of raw features, the classifier head, the training loop) and
varies only how node representations are built:

  conv="none"   each layer is a Linear on the node's own representation.
                No information crosses an edge. This is the control.
  conv="gcn"    GCNConv. With the defaults this is EdgeRiskGNN, bit for bit
                (asserted in tests/test_models.py).
  conv="sage"   GraphSAGE, mean aggregation.
  conv="gat"    GAT, one attention head so the width matches.

Training runs over a list of graphs, merged into one disconnected graph, so
the same code covers the within-network protocol (one graph, split edges) and
the unseen-network protocol (train on some graphs, score others).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

CONVS = ("none", "gcn", "sage", "gat")


class EdgeClassifier(torch.nn.Module):
    def __init__(self, in_dim, conv="gcn", hidden=32, layers=3, dropout=0.3):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}, got {conv!r}")
        self.conv = conv
        self.dropout = dropout

        dims = [in_dim] + [hidden] * layers
        # Created before the classifier, in the same order EdgeRiskGNN creates
        # conv1..conv3, so parameter initialisation consumes the RNG identically.
        self.layers = torch.nn.ModuleList(
            self._layer(conv, d_in, d_out) for d_in, d_out in zip(dims, dims[1:])
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((hidden + in_dim) * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 2),
        )

    @staticmethod
    def _layer(conv, d_in, d_out):
        if conv == "none":
            return torch.nn.Linear(d_in, d_out)
        if conv == "gcn":
            return GCNConv(d_in, d_out)
        if conv == "sage":
            return SAGEConv(d_in, d_out)
        return GATConv(d_in, d_out, heads=1)

    def forward(self, x, mp_edge_index, target_edge_index):
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h) if self.conv == "none" else layer(h, mp_edge_index)
            h = F.relu(h)
            if i < last:  # EdgeRiskGNN applies no dropout after its final conv
                h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.cat([h, x], dim=1)
        src, tgt = target_edge_index[0], target_edge_index[1]
        return self.classifier(torch.cat([h[src], h[tgt]], dim=1))


def merge_graphs(graphs):
    """Stack PyG graphs into one disconnected graph, offsetting node indices.

    Each graph needs x, y, edge_index (the edges to classify), mp_edge_index
    (the edges messages may flow along) and train_mask. A single graph passes
    through with its indices unchanged.
    """
    xs, ys, targets, mps, masks = [], [], [], [], []
    offset = 0
    for g in graphs:
        xs.append(g.x)
        ys.append(g.y)
        targets.append(g.edge_index + offset)
        mps.append(g.mp_edge_index + offset)
        masks.append(g.train_mask)
        offset += g.x.size(0)
    return (torch.cat(xs), torch.cat(ys), torch.cat(targets, dim=1),
            torch.cat(mps, dim=1), torch.cat(masks))


def train_edge_model(conv, graphs, epochs=400, lr=0.01, seed=42, **model_kwargs):
    """Train one EdgeClassifier over the training edges of every graph given.

    Seeding, optimiser, weight decay and class weighting follow train_gnn in
    gnn.py exactly, so conv="gcn" on one graph reproduces the published model.
    """
    x, y, edge_index, mp_edge_index, train_mask = merge_graphs(graphs)

    torch.manual_seed(seed)
    model = EdgeClassifier(in_dim=x.size(1), conv=conv, **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    counts = torch.bincount(y[train_mask], minlength=2).float()
    weight = counts.sum() / (2.0 * counts.clamp(min=1))

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, mp_edge_index, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=weight)
        loss.backward()
        opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, graph):
    """P(edge is on an optimal route), for every edge in graph.edge_index."""
    model.eval()
    out = model(graph.x, graph.mp_edge_index, graph.edge_index)
    return torch.softmax(out, dim=1)[:, 1]
