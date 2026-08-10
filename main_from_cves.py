"""
Run the attack-path-modeler pipeline from hand-typed CVEs instead of a Nessus
scan file. Real CVSS scores are pulled live from the NVD API (cached to
data/.cve_cache.json so repeat runs are instant).

Usage:
    python main_from_cves.py                    # uses data/known_hosts.json
    python main_from_cves.py path/to/hosts.json  # or your own spec
    python main_from_cves.py --no-serve          # skip auto-opening the dashboard

See data/known_hosts.json for the expected format — mark whichever host
bridges network segments (firewall, VPN gateway, jump host) with
"role": "gateway".
"""
import argparse
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # so this runs correctly regardless of cwd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which can't print — em dashes

from src.cve_lookup import build_hosts_from_known_cves
from src.graph import build_graph
from src.analysis import find_choke_points, find_highest_risk_path
from src.export import export_graph
from src.serve import serve_dashboard

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("spec_path", nargs="?", default="data/known_hosts.json",
                     help="path to a known-CVE host spec (default: data/known_hosts.json)")
parser.add_argument("--no-serve", action="store_true",
                     help="just export data/graph.json, don't start a server or open a browser")
args = parser.parse_args()

with open(args.spec_path, encoding="utf-8") as f:
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

risk = find_highest_risk_path(G)
risk_path_ips = risk["path"] if risk else []
if risk:
    path_str = " → ".join(G.nodes[n]["hostname"] for n in risk["path"])
    print(f"\nHighest risk path ({risk['hops']} hop{'s' if risk['hops'] != 1 else ''}): {path_str}")
else:
    print("\nNo multi-host path found — need at least 2 connected hosts to chain an attack path.")

export_graph(G, "data/graph.json", risk_path=risk_path_ips, source_label=f"known CVEs — {args.spec_path}")

if args.no_serve:
    print("\nRun `python -m http.server 8080` and open dashboard.html to view this graph.")
else:
    serve_dashboard()
