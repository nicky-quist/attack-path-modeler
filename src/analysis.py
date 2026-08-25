"""
Path ranking and choke-point analysis.

Two corrections to the original implementation:

*Choke points.* It used plain all-pairs betweenness centrality, which asks "how
many shortest paths between arbitrary host pairs run through this node?" On an
attack graph that is close to meaningless: hosts inside one VLAN reach each
other directly, so nothing routes through anyone, and on the old complete-graph
model exactly one node out of a hundred scored above zero. The question worth
asking is narrower — how many *entry-point-to-crown-jewel* paths run through
this node — which is what betweenness_centrality_subset computes.

*Highest-risk path.* It enumerated all simple paths between every ordered pair
of nodes, which is combinatorial and does not terminate in useful time on a
real estate. It also ranked by hop count first, so a four-hop crawl outranked a
one-hop compromise of a database. Because edge weight is now -log(P(exploit)),
the minimum-weight path is by construction the most probable attack chain, and
Dijkstra finds it in O(E log V).
"""
import networkx as nx

from src.graph import path_probability
from src.segmentation import INTERNET


def _sources_and_targets(G, policy):
    if G.number_of_nodes() == 0:
        return [], []
    sources = [s for s in policy.entry_points if s in G]
    if not sources:
        sources = [n for n in G if G.in_degree(n) == 0] or [next(iter(G))]
    targets = [t for t in policy.crown_jewels if t in G]
    if not targets:
        from src.labels import infer_crown_jewels
        targets = infer_crown_jewels(G, policy)
    return sources, targets


def top_attack_paths(G, policy, k=40, max_hops=6):
    """The k most probable entry -> crown-jewel chains across every asset.

    Shared by choke-point scoring and by the GNN's structural label, so both
    are defined against exactly the same notion of "a path that matters".
    """
    sources, targets = _sources_and_targets(G, policy)
    paths = []
    for src in sources:
        for tgt in targets:
            if src == tgt or tgt not in G:
                continue
            try:
                for i, path in enumerate(nx.shortest_simple_paths(G, src, tgt, weight="weight")):
                    if i >= k or len(path) - 1 > max_hops:
                        break
                    cost = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
                    paths.append({"path": path, "cost": round(cost, 4),
                                  "probability": round(path_probability(G, path), 6),
                                  "hops": len(path) - 1})
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    paths.sort(key=lambda r: r["cost"])
    return paths


def find_choke_points(G, policy, top=10, k=40):
    """Rank hosts by the share of top-k attack chains that traverse them.

    Reads directly as "containing this host breaks N of the M most probable
    routes to your crown jewels", which is what a defender is actually deciding
    between. Plain betweenness cannot say that: it only counts strictly
    shortest paths, so on this graph it scores three hosts above zero and the
    rest at exactly zero, which ranks nothing.
    """
    sources, targets = _sources_and_targets(G, policy)
    endpoints = set(sources) | set(targets) | {INTERNET}
    paths = top_attack_paths(G, policy, k=k)
    if not paths:
        return []

    counts = {}
    for entry in paths:
        for node in set(entry["path"]):
            if node in endpoints:
                continue
            counts[node] = counts.get(node, 0) + 1

    total = len(paths)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [(n, c / total, c, total) for n, c in ranked[:top]]


def betweenness_choke_points(G, policy, top=10):
    """Entry -> crown-jewel subset betweenness, kept for comparison."""
    sources, targets = _sources_and_targets(G, policy)
    if not sources or not targets:
        return []
    centrality = nx.betweenness_centrality_subset(
        G, sources=sources, targets=targets, normalized=True, weight="weight"
    )
    for n in set(sources) | set(targets) | {INTERNET}:
        centrality.pop(n, None)
    return sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top]


def most_probable_path(G, policy):
    """The single most probable entry -> crown-jewel chain in the graph.

    Minimum cumulative -log(p) == maximum ∏ p, so this is the chain an attacker
    is most likely to complete, not merely the shortest one.
    """
    sources, targets = _sources_and_targets(G, policy)
    if not sources or not targets:
        return None
    best = None
    for src in sources:
        try:
            costs, paths = nx.single_source_dijkstra(G, src, weight="weight")
        except nx.NodeNotFound:
            continue
        for tgt in targets:
            if tgt not in costs or tgt == src:
                continue
            cand = {
                "path": paths[tgt],
                "cost": round(costs[tgt], 4),
                "probability": round(path_probability(G, paths[tgt]), 6),
                "hops": len(paths[tgt]) - 1,
            }
            if best is None or cand["cost"] < best["cost"]:
                best = cand
    return best


def rank_attack_paths(G, policy, k=10):
    """The k most probable entry -> crown-jewel chains, best first."""
    sources, targets = _sources_and_targets(G, policy)
    ranked = []
    for src in sources:
        for tgt in targets:
            if src == tgt:
                continue
            try:
                for i, path in enumerate(nx.shortest_simple_paths(G, src, tgt, weight="weight")):
                    if i >= k:
                        break
                    cost = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
                    ranked.append({
                        "path": path,
                        "cost": round(cost, 4),
                        "probability": round(path_probability(G, path), 6),
                        "hops": len(path) - 1,
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    ranked.sort(key=lambda r: r["cost"])
    return ranked[:k]


def find_attack_paths(G, source, target, k=10):
    """Backwards-compatible helper: the k most probable source -> target chains."""
    ranked = []
    try:
        for i, path in enumerate(nx.shortest_simple_paths(G, source, target, weight="weight")):
            if i >= k:
                break
            cost = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))
            ranked.append({
                "path": path,
                "cost": round(cost, 4),
                "probability": round(path_probability(G, path), 6),
            })
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"shortest": [], "ranked": []}

    return {"shortest": ranked[0]["path"] if ranked else [], "ranked": ranked}
