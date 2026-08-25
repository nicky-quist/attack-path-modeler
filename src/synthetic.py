"""
A synthetic enterprise network for training and evaluation.

The original generator placed 100 hosts across two flat /24s and let the graph
builder join everything to everything, producing a complete digraph in which
betweenness centrality was zero for 99 of 100 nodes. This one models a
segmented estate and two facts the flat model ignored:

  * User workstations live in per-department VLANs that cannot route to each
    other. Reaching another department means going through the server tier.
  * Most hosts are patched. Only a minority expose an exploitable service, and
    a host with nothing exploitable has no inbound attack edges at all.

Together those give a sparse graph with a real server-tier bottleneck between
users and data, which is what makes choke-point analysis and a structural edge
label meaningful.

The CVEs here are fabricated, so EPSS has no score for them and exploitability
falls back to the CVSS-derived estimate documented in exploitability.py. Any
number produced from this data is a statement about the model, not about a real
network.
"""
import ipaddress
import random

from src.segmentation import INTERNET, Policy

# Business value per zone, as an asset inventory would record it. This is an
# input a real attack-path tool has and the model is entitled to use: knowing
# *where* the valuable systems are does not tell you which edges lie on an
# optimal route to them — that still requires the whole topology.
CRITICALITY = {"dmz": 0.3, "user-a": 0.2, "user-b": 0.2, "user-c": 0.2, "servers": 0.6, "data": 0.9}

ZONES = {
    "dmz":     {"cidr": "10.0.0.0/24", "count": 8,  "services": [(443, "https"), (8080, "http")]},
    "user-a":  {"cidr": "10.0.10.0/24", "count": 14, "services": [(3389, "rdp"), (445, "smb"), (22, "ssh")]},
    "user-b":  {"cidr": "10.0.11.0/24", "count": 14, "services": [(3389, "rdp"), (445, "smb")]},
    "user-c":  {"cidr": "10.0.12.0/24", "count": 12, "services": [(3389, "rdp"), (445, "smb"), (22, "ssh")]},
    "servers": {"cidr": "10.0.20.0/24", "count": 16, "services": [(445, "smb"), (5985, "winrm"), (22, "ssh")]},
    "data":    {"cidr": "10.0.30.0/24", "count": 6,  "services": [(3306, "mysql"), (1433, "mssql")]},
}

USER_ZONES = ["user-a", "user-b", "user-c"]

# Deliberately layered. No user VLAN reaches another, and none reaches `data`
# directly — every route from a workstation to a database traverses the server
# tier. That single constraint is what puts load-bearing structure in the graph.
def _rules():
    rules = [
        (INTERNET, "dmz", [443, 8080]),
        ("dmz", "dmz", [443, 8080]),
        ("dmz", "servers", [445, 5985]),
        ("servers", "servers", [445, 5985, 22]),
        ("servers", "data", [3306, 1433]),
        ("data", "data", [3306]),
    ]
    for uz in USER_ZONES:
        rules.append((uz, uz, [3389, 445, 22]))     # lateral movement inside a department
        rules.append((uz, "servers", [445, 5985]))  # everyone needs the app tier
    return rules


def build_policy(crown_jewels):
    zones = {name: [ipaddress.ip_network(z["cidr"])] for name, z in ZONES.items()}
    rules = {}
    for src, dst, ports in _rules():
        rules.setdefault((src, dst), set()).update(ports)
    return Policy(zones=zones, rules=rules, entry_points=[INTERNET], crown_jewels=crown_jewels)


def generate_synthetic_network(seed=42, vuln_rate=0.45):
    """Return (hosts, policy) for a segmented synthetic estate.

    `vuln_rate` is the share of hosts carrying an exploitable service. The rest
    are modelled as patched: they still exist as nodes and can originate an
    attack once compromised, but nothing can be exploited *on* them.
    """
    rng = random.Random(seed)
    hosts = []

    for zone_name, spec in ZONES.items():
        base = ipaddress.ip_network(spec["cidr"]).network_address
        for i in range(spec["count"]):
            ip = str(ipaddress.ip_address(int(base) + i + 1))
            vulns = []

            if rng.random() < vuln_rate:
                neglected = rng.random() < 0.3
                for _ in range(rng.randint(2, 3) if neglected else 1):
                    port, service = rng.choice(spec["services"])
                    cvss = round(rng.uniform(7.5, 10.0), 1) if neglected else round(rng.uniform(3.0, 7.4), 1)
                    vulns.append({
                        "cve": f"CVE-{rng.randint(2018, 2025)}-{rng.randint(10000, 99999)}",
                        "cvss": cvss,
                        "port": str(port),
                        "service": service,
                    })

            hosts.append({
                "ip": ip,
                "hostname": f"{zone_name}-{i + 1:02d}",
                "vulns": vulns,
                "zone": zone_name,
                "criticality": CRITICALITY[zone_name],
            })

    # The crown jewels are the databases holding the things worth stealing.
    crown_jewels = [h["ip"] for h in hosts if h["zone"] == "data"]
    return hosts, build_policy(crown_jewels)
