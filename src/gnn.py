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

    edge_index = []
    edge_labels = []
    for src, tgt, data in G.edges(data=True):
        edge_index.append([node_index[src], node_index[tgt]])
        label = 1 if G.nodes[tgt]["max_cvss"] > 8.5 else 0
        edge_labels.append(label)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(edge_labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class EdgeRiskGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 16)
        self.classifier = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
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

def train_gnn(data):
    model = EdgeRiskGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        weight = torch.tensor([1.0, 2.5])
        loss = F.cross_entropy(out, data.y, weight=weight)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    
    correct = (pred == data.y).sum().item()
    total = len(data.y)
    high_risk_predicted = (pred == 1).sum().item()
    high_risk_actual = (data.y == 1).sum().item()
    
    print(f"\nAccuracy:          {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Actual high risk:  {high_risk_actual}/{total}")
    print(f"Predicted high risk: {high_risk_predicted}/{total}")

    return model