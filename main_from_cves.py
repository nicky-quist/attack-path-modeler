"""
Run the attack-path-modeler pipeline from hand-typed CVEs instead of a Nessus
scan file. Real CVSS scores are pulled live from the NVD API (cached to
data/.cve_cache.json so repeat runs are instant).

Usage:
    python main_from_cves.py [path/to/hosts.json]

Defaults to data/known_hosts.json. See that file for the expected format —
mark whichever host bridges network segments (firewall, VPN gateway, jump
host) with "role": "gateway".
"""
import sys
import json

from src.cve_lookup import build_hosts_from_known_cves
from src.graph import build_graph
from src.analysis import find_choke_points
from src.export import export_graph

spec_path = sys.argv[1] if len(sys.argv) > 1 else "data/known_hosts.json"

with open(spec_path) as f:
    spec = json.load(f)

num_cves = sum(len(h.get("cves", [])) for h in spec.values())
print(f"Looking up {num_cves} CVE(s) from NVD (cached lookups are instant, "
      f"new ones are rate-limited to ~1 every 6s)...")
hosts = build_hosts_from_known_cves(spec)

print("\nHosts:")
for h in hosts:
    max_cvss = max((v["cvss"] for v in h["vulns"]), default=0)
    role = f" [{h['role']}]" if "role" in h else ""
    cves = ", ".join(v["cve"] for v in h["vulns"])
    print(f"  {h['hostname']}{role} ({h['ip']}) — max CVSS {max_cvss} — {cves}")

G = build_graph(hosts)
print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\nChoke points (betweenness centrality):")
for ip, score in find_choke_points(G)[:5]:
    print(f"  {G.nodes[ip]['hostname']} ({ip}): {score:.3f}")

export_graph(G, "data/graph.json")
print("\nOpen dashboard.html (via `python -m http.server 8080`) to visualize this graph.")
