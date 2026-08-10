import json
from datetime import datetime, timezone
from src.parser import parse_nessus
from src.graph import build_graph

def export_graph(G, filepath, risk_path=None, source_label=None):

    # Build the nodes list — one dict per host
    nodes = []
    for node_id in G.nodes:
        node = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "hostname": node["hostname"],
            "max_cvss": node["max_cvss"]
        })

    # Build the links list — one dict per edge (connection between hosts)
    # G.edges(data=True) gives us (source_ip, target_ip, edge_attributes)
    links = []
    for source, target, data in G.edges(data=True):
        links.append({
            "source": source,
            "target": target,
            "weight": data["weight"],
            "cve": data["cve"],
            "service": data["service"],
            "port": data["port"]
        })

    # Combine into one object D3 can read.
    # riskPath/sourceLabel/generatedAt let the dashboard show the highest-risk
    # chain and which dataset it's currently displaying, instead of guessing.
    graph_data = {
        "nodes": nodes,
        "links": links,
        "riskPath": risk_path or [],
        "sourceLabel": source_label or "",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Write it to a file
    # json.dump converts the Python dict to JSON text and saves it
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph exported to {filepath}")


if __name__ == "__main__":
    hosts = parse_nessus("data/sample.nessus")
    G = build_graph(hosts)
    export_graph(G, "data/graph.json")