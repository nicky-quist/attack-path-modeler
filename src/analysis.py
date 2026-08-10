import networkx as nx
from src.parser import parse_nessus
from src.graph import build_graph

def find_attack_paths(G, source, target):

    shortest = nx.shortest_path(G, source=source, target=target, weight="weight")

    all_paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=4))

    scored_paths = []
    for path in all_paths:
        total_weight = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
        scored_paths.append({"path": path, "score": round(total_weight, 4)})

    scored_paths.sort(key=lambda x: x["score"])

    return {
        "shortest": shortest,
        "all_paths": all_paths,
        "ranked": scored_paths
    }

def find_choke_points(G):
    centrality = nx.betweenness_centrality(G, weight="weight")
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return ranked


def find_highest_risk_path(G, max_hops=4):
    """Find the single most dangerous chain in the whole graph, with no
    source/target known in advance — used when hosts come from typed CVEs
    rather than a fixed demo dataset.

    Ranks by hop count first (deepest lateral-movement chain), then by
    lowest cumulative weight as a tiebreaker (easiest to actually traverse).
    A short, low-weight path is a weaker story than a longer chain that
    still represents a real, walkable route through the network.
    """
    best = None
    for source in G.nodes:
        for target in G.nodes:
            if source == target:
                continue
            for path in nx.all_simple_paths(G, source, target, cutoff=max_hops):
                weight = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
                hops = len(path) - 1
                score = round(weight, 4)
                if best is None or (hops, -score) > (best["hops"], -best["score"]):
                    best = {"path": path, "hops": hops, "score": score}
    return best


if __name__ == "__main__":
    hosts = parse_nessus("data/sample.nessus")
    G = build_graph(hosts)
    
    results = find_attack_paths(G, "10.0.0.1", "10.0.0.3")
    print("Shortest path:", results["shortest"])
    print("All paths:", results["all_paths"])
    print("Ranked paths:", results["ranked"])
    
    print("\nChoke points:", find_choke_points(G))