"""
The models the GNN has to beat.

A headline number with nothing to compare it against says nothing. The original
project reported 100% precision and recall on the old label without noting that
a one-line threshold rule scored exactly the same — which is the fact that
made the result meaningless. These baselines exist so that can't happen again,
and they are reported next to the GNN every time it trains.

  majority            predict the commoner class. The floor: any model that
                      cannot beat this has learned nothing.
  cvss-threshold      the ORIGINAL label rule, `target.max_cvss > 8.5`, kept
                      deliberately. It scored 100% against the old target
                      because the target was a copy of it. Against a structural
                      label it should collapse, and that collapse is the
                      evidence the leak is gone.
  logistic-regression a linear model over the same node features the GNN gets,
                      for both endpoints, with no access to graph structure
                      beyond each endpoint's own degree. This is the real
                      competitor: it isolates how much of the task needs
                      message passing rather than local features.
"""
import torch

from src.metrics import binary_metrics


def majority_baseline(y_train, y_test):
    cls = int(torch.mode(y_train).values)
    pred = torch.full_like(y_test, cls)
    return binary_metrics(pred, y_test)


def cvss_threshold_baseline(G, edges, y_test, test_idx, threshold=8.5):
    """The original label rule, applied as a predictor."""
    pred = torch.tensor(
        [1 if G.nodes[edges[i][1]]["max_cvss"] > threshold else 0 for i in test_idx],
        dtype=torch.long,
    )
    return binary_metrics(pred, y_test)


def logistic_regression_baseline(edge_x, y, train_mask, test_mask, epochs=400, lr=0.05):
    """Linear model on [src_features ‖ tgt_features]. No message passing."""
    torch.manual_seed(0)
    model = torch.nn.Linear(edge_x.size(1), 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    counts = torch.bincount(y[train_mask], minlength=2).float()
    weight = counts.sum() / (2.0 * counts.clamp(min=1))

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(edge_x[train_mask]), y[train_mask], weight=weight
        )
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(edge_x).argmax(dim=1)
    return binary_metrics(pred[test_mask], y[test_mask])
