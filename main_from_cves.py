"""
Run the attack-path-modeler pipeline from hand-typed CVEs instead of a Nessus
scan file. Real CVSS scores are pulled live from the NVD API (cached to
data/.cve_cache.json so repeat runs are instant).

Usage:
    python main_from_cves.py                    # uses data/known_hosts.json
    python main_from_cves.py path/to/hosts.json  # or your own spec
    python main_from_cves.py --interactive       # type hosts/CVEs at the prompt, no file needed
    python main_from_cves.py --no-serve          # skip auto-opening the dashboard

See data/known_hosts.json for the expected format — mark whichever host
bridges network segments (firewall, VPN gateway, jump host) with
"role": "gateway". Prefer a browser? Run this once, then open builder.html
from the dashboard for the same thing as a form instead of a prompt.
"""
import argparse
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # so this runs correctly regardless of cwd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which can't print — em dashes

from src.analysis import find_choke_points, most_probable_path
from src.cve_lookup import build_hosts_from_known_cves
from src.exploitability import annotate_hosts
from src.export import export_graph
from src.graph import build_graph
from src.segmentation import resolve_policy
from src.serve import serve_dashboard


def prompt_for_hosts():
    print("=== Add hosts (leave hostname blank when you're done) ===\n")
    spec = {}
    while True:
        hostname = input("Hostname: ").strip()
        if not hostname:
            break

        ip = input("  IP address: ").strip()
        while not ip:
            ip = input("  IP address (required): ").strip()

        cves_raw = input("  CVE IDs (comma-separated): ").strip()
        while not cves_raw:
            cves_raw = input("  CVE IDs (required, comma-separated): ").strip()
        cves = [c.strip() for c in cves_raw.split(",") if c.strip()]

        port = input("  Port (optional): ").strip()
        service = input("  Service (optional): ").strip()
        is_gateway = input("  Gateway? bridges network segments (y/N): ").strip().lower() == "y"

        spec[hostname] = {"ip": ip, "cves": cves, "port": port, "service": service}
        if is_gateway:
            spec[hostname]["role"] = "gateway"
        print()

    return spec


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("spec_path", nargs="?", default="data/known_hosts.json",
                     help="path to a known-CVE host spec (default: data/known_hosts.json)")
parser.add_argument("--interactive", "-i", action="store_true",
                     help="type hosts/CVEs at the prompt instead of reading a spec file")
parser.add_argument("--no-serve", action="store_true",
                     help="just export data/graph.json, don't start a server or open a browser")
parser.add_argument("--policy", default="data/segmentation.json",
                     help="segmentation policy to apply (default: data/segmentation.json; "
                          "one zone per /24 is synthesised if the file is absent)")
parser.add_argument("--offline", action="store_true",
                     help="skip EPSS/KEV lookups and estimate exploitability from CVSS")
args = parser.parse_args()

if args.interactive:
    spec = prompt_for_hosts()
    if not spec:
        print("No hosts entered — nothing to do.")
        sys.exit(0)
else:
    with open(args.spec_path, encoding="utf-8") as f:
        spec = json.load(f)

num_cves = sum(len(h.get("cves", [])) for h in spec.values())
print(f"Looking up {num_cves} CVE(s) from NVD (cached lookups are instant, "
      f"new ones are rate-limited to ~1 every 6s)...")
hosts = build_hosts_from_known_cves(spec)

print("Resolving exploitability" + (" (offline — CVSS estimates)" if args.offline
                                    else " (EPSS + CISA KEV)") + "...")
hosts = annotate_hosts(hosts, offline=args.offline)

print("\nHosts:")
for h in hosts:
    max_cvss = max((v["cvss"] for v in h["vulns"]), default=0)
    max_p = max((v.get("p_exploit", 0) for v in h["vulns"]), default=0)
    role = f" [{h['role']}]" if "role" in h else ""
    kev = " [KEV]" if any(v.get("in_kev") for v in h["vulns"]) else ""
    cves = ", ".join(v["cve"] for v in h["vulns"])
    print(f"  {h['hostname']}{role}{kev} ({h['ip']}) — max CVSS {max_cvss}, "
          f"P(exploit) {max_p:.4f} — {cves}")

policy = resolve_policy(hosts, args.policy)
G = build_graph(hosts, policy)
_n, _e = G.number_of_nodes(), G.number_of_edges()
_density = _e / (_n * (_n - 1)) if _n > 1 else 0
print(f"\nGraph: {_n} nodes, {_e} edges (density {_density:.3f})")

chokes = find_choke_points(G, policy, top=5)
if chokes:
    print("\nChoke points — share of top attack chains passing through:")
    for ip, share, count, total in chokes:
        print(f"  {G.nodes[ip]['hostname']} ({ip}): {share:.1%} ({count}/{total} chains)")

risk = most_probable_path(G, policy)
risk_path_ips = risk["path"] if risk else []
if risk:
    path_str = " → ".join(G.nodes[x]["hostname"] for x in risk["path"])
    print(f"\nMost probable chain ({risk['hops']} hop{'s' if risk['hops'] != 1 else ''}): {path_str}")
    print(f"  P(success) = {risk['probability']:.4f}")
else:
    print("\nNo multi-host path found — need at least 2 connected hosts to chain an attack path.")

source_label = "known CVEs — typed in interactively" if args.interactive else f"known CVEs — {args.spec_path}"
export_graph(G, "data/graph.json", risk_path=risk_path_ips, source_label=source_label,
             path_probability=risk["probability"] if risk else None,
             choke_points=[{"id": c[0], "share": round(c[1], 4)} for c in chokes])

if args.no_serve:
    print("\nRun `python -m http.server 8080` and open dashboard.html to view this graph.")
else:
    serve_dashboard()
