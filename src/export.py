import json
from src.parser import parse_nessus
from src.graph import build_graph

def export_graph(G, filepath):
    
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

    # Combine into one object D3 can read
    graph_data = {
        "nodes": nodes,
        "links": links
    }

    # Write it to a file
    # json.dump converts the Python dict to JSON text and saves it
    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph exported to {filepath}")


if __name__ == "__main__":
    hosts = parse_nessus("data/sample.nessus")
    G = build_graph(hosts)
    export_graph(G, "data/graph.json")