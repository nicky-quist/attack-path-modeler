"""
The prediction target for the GNN.

The original label was `1 if target.max_cvss > 8.5`, while `max_cvss` was
simultaneously node feature 0 and fed to the classifier raw through the skip
connection. The label was therefore a deterministic function of an input, and a
one-line threshold rule scores 100% on it with no model at all. Perfect
precision and recall were not evidence of learning; they were the only
achievable outcome, and no amount of real scan data would change that.

The replacement asks something a node cannot answer about itself:

    Does this edge lie on one of the K most probable attack chains
    from an internet-facing entry point to a crown-jewel asset?

That depends on where the edge sits in the whole graph relative to assets it
has no local knowledge of. A model must aggregate information over the
topology to predict it, which is precisely the thing message passing is for —
and it is why a node-feature-only baseline should now lose.
"""
import networkx as nx

from src.segmentation import INTERNET


def infer_crown_jewels(G, policy, count=None):
    """Pick target assets when the policy doesn't name any.

    Uses depth from the entry point: the hosts an attacker must work hardest to
    reach are the ones sitting behind the most segmentation, which is where the
    valuable things usually live.
    """
    sources = [s for s in policy.entry_points if s in G] or [n for n in G if n != INTERNET][:1]
    if not sources:
        return []

    # Scale with the estate. On a four-host spec, "the three deepest" is nearly
    # every host, and the cheapest of those is whatever sits closest to the
    # perimeter — the opposite of a crown jewel.
    if count is None:
        count = max(1, min(5, (G.number_of_nodes() - 1) // 10))

    # Depth in *hops*, not in -log(p). Weighted distance measures how hard a
    # host is to exploit, which is a different question: a brittle host one hop
    # from the perimeter can carry a higher weighted cost than a database three
    # segments deep, and picking by weight would call the brittle host the
    # crown jewel. Criticality breaks ties.
    depth = {}
    for src in sources:
        for node, hops in nx.single_source_shortest_path_length(G, src).items():
            if node == INTERNET or node in sources:
                continue
            depth[node] = max(depth.get(node, 0), hops)

    ranked = sorted(
        depth.items(),
        key=lambda kv: (kv[1], G.nodes[kv[0]].get("criticality", 0.5)),
        reverse=True,
    )
    return [n for n, _ in ranked[:count]]


def critical_path_edges(G, policy, k=8, max_hops=6):
    """Edges appearing on any of the k most probable entry -> crown-jewel chains."""
    targets = [t for t in policy.crown_jewels if t in G] or infer_crown_jewels(G, policy)
    sources = [s for s in policy.entry_points if s in G]
    if not sources:
        sources = [n for n in G if G.in_degree(n) == 0] or [next(iter(G))]

    critical = set()
    for src in sources:
        for tgt in targets:
            if src == tgt or tgt not in G:
                continue
            try:
                for i, path in enumerate(nx.shortest_simple_paths(G, src, tgt, weight="weight")):
                    if i >= k or len(path) - 1 > max_hops:
                        break
                    critical.update(zip(path, path[1:]))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    return critical

def distance_to_crown_jewels(G, policy):
    """d(v) = cost of the cheapest route from v to any crown jewel.

    Computed by running Dijkstra backwards from each asset over the reversed
    graph, so one pass per asset covers every node.
    """
    targets = [t for t in policy.crown_jewels if t in G] or infer_crown_jewels(G, policy)
    R = G.reverse(copy=False)
    dist = {}
    for tgt in targets:
        try:
            lengths = nx.single_source_dijkstra_path_length(R, tgt, weight="weight")
        except nx.NodeNotFound:
            continue
        for node, d in lengths.items():
            if node not in dist or d < dist[node]:
                dist[node] = d
    return dist


def label_edges(G, policy, tolerance=0.1):
    """Label each edge: does taking it advance the attacker optimally?

    For edge (u, v), positive when

        w(u, v) + d(v)  <=  d(u) + tolerance

    i.e. moving to v costs about as little as the best move available from u.
    This is the Bellman optimality condition on the attack graph: it marks the
    edges an optimal — or near-optimal — attacker would actually traverse on
    the way to a crown jewel.

    It cannot be answered from a node's own features. d(u) and d(v) are global
    quantities defined by the entire downstream topology and the location of
    assets the node has no local knowledge of, so predicting this requires
    aggregating information across the graph. That is the whole justification
    for using a GNN here, and it is exactly what the previous
    `max_cvss > 8.5` label failed to require.

    `tolerance` is in units of -log(p); 0.1 admits moves up to ~10% less
    probable than the best one, which keeps the label from being brittle to
    ties and floating-point noise.
    """
    dist = distance_to_crown_jewels(G, policy)
    edges = list(G.edges())
    labels = []
    for u, v in edges:
        du, dv = dist.get(u), dist.get(v)
        if du is None or dv is None:
            labels.append(0)
            continue
        w = G[u][v]["weight"]
        labels.append(1 if (w + dv) <= (du + tolerance) else 0)
    return edges, labels
