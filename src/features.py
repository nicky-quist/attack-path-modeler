"""
Node features, shared by the GNN and every baseline so the comparison is fair.

Deliberately excluded: anything derived from distance-to-crown-jewel, and the
`is_crown_jewel` flag itself. Those are what the label is computed from, and
handing them to the model would recreate exactly the leak that made the
original `max_cvss > 8.5` target meaningless.

`criticality` IS included. Asset value comes from an inventory, not from the
graph, and knowing which systems are valuable does not tell you which edges lie
on an optimal route to them.
"""
import torch

FEATURE_NAMES = [
    "max_cvss",         # severity of the worst vuln on the host
    "num_vulns",        # how much is wrong with it
    "num_open_ports",   # exposed surface
    "max_p_exploit",    # EPSS / KEV-informed probability of exploitation
    "in_kev",           # is anything on this host actively exploited in the wild
    "criticality",      # business value, from the asset inventory
    "out_degree",       # local structure: hosts it can attack
    "in_degree",        # local structure: hosts that can attack it
]


def node_feature_matrix(G, node_list):
    rows = []
    n = max(len(node_list) - 1, 1)
    for ip in node_list:
        node = G.nodes[ip]
        rows.append([
            node["max_cvss"] / 10.0,
            len(node["vulns"]),
            len(node["open_ports"]),
            node.get("max_p_exploit", 0.0),
            1.0 if node.get("in_kev") else 0.0,
            node.get("criticality", 0.5),
            G.out_degree(ip) / n,
            G.in_degree(ip) / n,
        ])
    x = torch.tensor(rows, dtype=torch.float)
    # Column-normalise: raw counts and probabilities live on very different
    # scales, which slows and destabilises training.
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)
