import json
from datetime import datetime, timezone
from src.parser import parse_nessus
from src.graph import build_graph

def export_graph(G, filepath, risk_path=None, source_label=None, path_probability=None, choke_points=None):

    # Build the nodes list — one dict per host
    nodes = []
    for node_id in G.nodes:
        node = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "hostname": node["hostname"],
            "max_cvss": node["max_cvss"],
            "zone": node.get("zone"),
            "max_p_exploit": round(node.get("max_p_exploit", 0.0), 4),
            "in_kev": bool(node.get("in_kev")),
            "criticality": round(node.get("criticality", 0.5), 2),
            "is_crown_jewel": bool(node.get("is_crown_jewel")),
            # The dashboard tooltip lists these; without them it silently
            # rendered an empty vulnerability list for every host.
            "vulns": [
                {
                    "cve": v.get("cve"),
                    "cvss": v.get("cvss"),
                    "service": v.get("service"),
                    "port": v.get("port"),
                    "p_exploit": round(v["p_exploit"], 4) if v.get("p_exploit") else None,
                    "in_kev": bool(v.get("in_kev")),
                }
                for v in node.get("vulns", [])
            ],
        })

    # Build the links list — one dict per edge (connection between hosts)
    # G.edges(data=True) gives us (source_ip, target_ip, edge_attributes)
    links = []
    for source, target, data in G.edges(data=True):
        links.append({
            "source": source,
            "target": target,
            "weight": data["weight"],
            "p_exploit": data.get("p_exploit"),
            "in_kev": bool(data.get("in_kev")),
            "p_source": data.get("p_source"),
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
        "riskPathProbability": path_probability,
        "chokePoints": choke_points or [],
        "sourceLabel": source_label or "",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Write it to a file
    # json.dump converts the Python dict to JSON text and saves it
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph exported to {filepath}")


if __name__ == "__main__":
    from src.segmentation import resolve_policy
    hosts = parse_nessus("data/sample.nessus")
    policy = resolve_policy(hosts, "data/segmentation.json")
    export_graph(build_graph(hosts, policy), "data/graph.json")