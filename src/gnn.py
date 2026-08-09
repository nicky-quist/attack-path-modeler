import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from src.parser import parse_nessus
from src.graph import build_graph

def build_pyg_data(hosts, G):
    node_list = list(G.nodes())
    node_index = {ip: i for i, ip in enumerate(node_list)}

    features = []
    for ip in node_list:
        node = G.nodes[ip]
        num_vulns = len(node["vulns"])
        max_cvss = node["max_cvss"]
        num_ports = len(set(v["port"] for v in node["vulns"]))
        is_internal = 1.0 if ip.startswith("10.0.1.") else 0.0
        features.append([max_cvss, num_vulns, num_ports, is_internal])

    x = torch.tensor(features, dtype=torch.float)
    # Normalize each feature column to [0, 1] — max_cvss (0-10) and num_vulns/num_ports
    # (small integer counts) were on different scales, which slows/destabilizes GCN training.
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    x = (x - x_min) / (x_max - x_min + 1e-8)

    edge_index = []
    edge_labels = []
    for src, tgt, data in G.edges(data=True):
        edge_index.append([node_index[src], node_index[tgt]])
        label = 1 if G.nodes[tgt]["max_cvss"] > 8.5 else 0
        edge_labels.append(label)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(edge_labels, dtype=torch.long)

    # Train/test split on edges so accuracy reflects generalization, not memorization.
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    split = int(num_edges * 0.8)
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[perm[:split]] = True
    test_mask = ~train_mask

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


class EdgeRiskGNN(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 16)
        # +4 for the raw node features, concatenated back in below (skip connection).
        self.classifier = torch.nn.Linear((16 + 4) * 2, 2)
        self.dropout = dropout

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # This graph is extremely dense (avg degree ~99 on 100 nodes), so two rounds of
        # GCN message-passing smooths each node's embedding into close to the graph-wide
        # average — the host's own max_cvss signal gets buried. Concatenating the raw
        # features back in gives the classifier a direct, un-smoothed view of each node.
        x = torch.cat([x, x0], dim=1)
        src = edge_index[0]
        tgt = edge_index[1]
        edge_features = torch.cat([x[src], x[tgt]], dim=1)
        return self.classifier(edge_features)

def generate_synthetic_data(num_hosts=100):
    import random
    random.seed(42)
    hosts = []
    for i in range(num_hosts):
        subnet = "10.0.0" if i < num_hosts // 2 else "10.0.1"
        ip = f"{subnet}.{i % 50 + 1}"
        is_critical = random.random() < 0.4
        num_vulns = random.randint(3, 6) if is_critical else random.randint(1, 2)
        vulns = []
        for _ in range(num_vulns):
            cvss = round(random.uniform(8.5, 10.0), 1) if is_critical else round(random.uniform(3.0, 7.0), 1)
            vulns.append({
                "cve": f"CVE-{random.randint(2015,2023)}-{random.randint(1000,9999)}",
                "cvss": cvss,
                "port": str(random.choice([22, 80, 443, 3306, 3389, 445, 8080])),
                "service": random.choice(["ssh", "http", "https", "mysql", "rdp", "smb"])
            })
        hosts.append({"ip": ip, "hostname": f"host-{i}", "vulns": vulns})
    return hosts

def _class_report(name, pred, y, mask):
    pred, y = pred[mask], y[mask]
    total = len(y)
    correct = (pred == y).sum().item()
    actual_high = (y == 1).sum().item()
    pred_high = (pred == 1).sum().item()
    true_pos = ((pred == 1) & (y == 1)).sum().item()
    precision = true_pos / pred_high if pred_high else 0.0
    recall = true_pos / actual_high if actual_high else 0.0
    print(f"{name:5s} | acc {correct}/{total} ({correct/total*100:5.1f}%) "
          f"| actual-high {actual_high:3d} pred-high {pred_high:3d} "
          f"| precision {precision:.2f} recall {recall:.2f}")

def train_gnn(data):
    model = EdgeRiskGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Weight classes by their actual inverse frequency in the TRAIN split, not a guessed
    # constant — an arbitrary weight (e.g. 2.5) can make "always predict class 1" the
    # cheapest solution for the optimizer to collapse into, which is what was happening.
    train_y = data.y[data.train_mask]
    class_counts = torch.bincount(train_y, minlength=2).float()
    class_weight = class_counts.sum() / (2.0 * class_counts.clamp(min=1))

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
            _class_report("train", pred, data.y, data.train_mask)
            _class_report("test ", pred, data.y, data.test_mask)
            print()

    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)

    print("Final:")
    _class_report("train", pred, data.y, data.train_mask)
    _class_report("test ", pred, data.y, data.test_mask)

    return model
