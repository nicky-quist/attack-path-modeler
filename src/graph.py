"""
Builds the attack graph.

Two things changed from the original co-location model, and both matter:

1. An edge now requires *reachability to a vulnerable service*, not merely two
   hosts sharing a subnet. A -> B exists only where segmentation policy permits
   A's zone to reach a port on B, and B has a vulnerability on that port. This
   is what stops the graph collapsing into a complete digraph.

2. Edge weight is -log(P(exploit)) instead of 1/CVSS. Summing 1/CVSS along a
   path produces a number that is not a quantity of anything. Summing
   -log(p) gives -log(∏ p), so the minimum-weight path found by Dijkstra is
   exactly the *most probable* attack chain, and exp(-cost) is that chain's
   end-to-end success probability.
"""
import math

import networkx as nx

from src.exploitability import cvss_fallback_probability
from src.segmentation import INTERNET, resolve_policy


def _p_exploit(vuln):
    """P(exploit) for a vuln, falling back to a CVSS estimate if the
    exploitability annotations were never fetched."""
    p = vuln.get("p_exploit")
    if p is None:
        p = cvss_fallback_probability(vuln.get("cvss", 0))
    return max(1e-4, min(0.999, float(p)))


def edge_cost(p):
    """-log(p): additive along a path, and monotone decreasing in probability."""
    return round(-math.log(p), 4)


def build_graph(hosts, policy=None, include_internet=True):
    """Build a directed attack graph from parsed hosts.

    `policy` is a segmentation.Policy; when omitted, one is synthesised from the
    hosts (one zone per /24, gateways bridging outward).
    """
    if policy is None:
        policy = resolve_policy(hosts)

    G = nx.DiGraph()

    for host in hosts:
        vulns = host["vulns"]
        G.add_node(
            host["ip"],
            hostname=host["hostname"],
            vulns=vulns,
            zone=policy.zone_of(host["ip"]),
            max_cvss=max((v["cvss"] for v in vulns), default=0),
            max_p_exploit=max((_p_exploit(v) for v in vulns), default=0.0),
            in_kev=any(v.get("in_kev") for v in vulns),
            open_ports=sorted({str(v.get("port")) for v in vulns if v.get("port")}),
            is_crown_jewel=host["ip"] in policy.crown_jewels,
            criticality=float(host["criticality"]) if "criticality" in host
            else policy.criticality_of(host["ip"]),
        )
        if host.get("role"):
            G.nodes[host["ip"]]["role"] = host["role"]

    if include_internet:
        G.add_node(
            INTERNET,
            hostname="internet",
            vulns=[],
            zone=INTERNET,
            max_cvss=0,
            max_p_exploit=0.0,
            in_kev=False,
            open_ports=[],
            is_crown_jewel=False,
            criticality=0.0,
        )

    def _best_reachable_vuln(src_zone, target):
        """The most exploitable vuln on `target` that `src_zone` can actually reach."""
        best = None
        for vuln in G.nodes[target]["vulns"]:
            if not policy.can_reach(src_zone, G.nodes[target]["zone"], vuln.get("port")):
                continue
            p = _p_exploit(vuln)
            if best is None or p > best[0]:
                best = (p, vuln)
        return best

    sources = list(G.nodes)
    for src in sources:
        src_zone = G.nodes[src]["zone"]
        for tgt in sources:
            if src == tgt or tgt == INTERNET:
                continue
            best = _best_reachable_vuln(src_zone, tgt)
            if best is None:
                continue
            p, vuln = best
            G.add_edge(
                src, tgt,
                weight=edge_cost(p),
                p_exploit=round(p, 6),
                cve=vuln.get("cve"),
                service=vuln.get("service"),
                port=vuln.get("port"),
                in_kev=bool(vuln.get("in_kev")),
                p_source=vuln.get("p_source", "cvss-estimate"),
            )

    # An internet node with no way in is noise in the visualisation and skews
    # centrality; drop it when policy exposes nothing to the outside.
    if include_internet and G.out_degree(INTERNET) == 0:
        G.remove_node(INTERNET)

    return G


def path_probability(G, path):
    """End-to-end success probability of a chain: ∏ p over its edges."""
    p = 1.0
    for a, b in zip(path, path[1:]):
        p *= G[a][b]["p_exploit"]
    return p
