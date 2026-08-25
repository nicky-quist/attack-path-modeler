"""
Network segmentation policy — which zones can reach which services.

The original graph joined every host to every other host in its subnet, which
produced a complete digraph (density 0.5 on the synthetic set). That has two
consequences worth naming: betweenness centrality goes to zero for all but the
one hand-placed bridge, because in a complete subgraph the shortest route
between any two hosts is the direct edge and nothing routes through anyone; and
message passing over a graph that dense smooths every node embedding toward the
graph-wide mean.

Real networks are not complete graphs. Reachability is governed by firewall and
VLAN policy, and an attacker needs a *service* to attack, not merely a
co-located host. This module models that: a zone map plus allow-rules over
ports, with `internet` as the virtual zone every external attacker starts in.
"""
import ipaddress
import json
import os

INTERNET = "internet"

# Used when no policy file is supplied. One zone per /24, the ports an attacker
# realistically pivots on inside a segment, and gateways bridging outward — so
# existing datasets keep working without a hand-written policy.
DEFAULT_LATERAL_PORTS = [22, 135, 139, 445, 3389, 5985, 5986]
DEFAULT_INGRESS_PORTS = [80, 443, 8080, 8443]


class Policy:
    def __init__(self, zones, rules, entry_points=None, crown_jewels=None, criticality=None):
        # zones: {zone_name: [ip_network, ...]}
        self.zones = zones
        # rules: {(from_zone, to_zone): set(ports)}
        self.rules = rules
        self.entry_points = entry_points or [INTERNET]
        self.crown_jewels = crown_jewels or []
        # Business value per zone, as an asset inventory would record it.
        self.criticality = criticality or {}

    def zone_of(self, ip):
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return None
        for name, nets in self.zones.items():
            for net in nets:
                if addr in net:
                    return name
        return None

    def criticality_of(self, ip):
        zone = self.zone_of(ip)
        if zone in self.criticality:
            return float(self.criticality[zone])
        return 0.9 if ip in self.crown_jewels else 0.5

    def allowed_ports(self, from_zone, to_zone):
        return self.rules.get((from_zone, to_zone), set())

    def can_reach(self, from_zone, to_zone, port):
        if from_zone is None or to_zone is None:
            return False
        try:
            port = int(port)
        except (TypeError, ValueError):
            return False
        return port in self.allowed_ports(from_zone, to_zone)

    def summary(self):
        lines = [f"zones: {', '.join(sorted(self.zones))}"]
        for (src, dst), ports in sorted(self.rules.items()):
            lines.append(f"  {src} -> {dst}: {sorted(ports)}")
        lines.append(f"entry points: {self.entry_points}")
        lines.append(f"crown jewels: {self.crown_jewels or '(none declared)'}")
        return "\n".join(lines)


def load_policy(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    zones = {}
    for name, cidrs in spec.get("zones", {}).items():
        if isinstance(cidrs, dict):
            cidrs = cidrs.get("cidrs", [])
        zones[name] = [ipaddress.ip_network(c, strict=False) for c in cidrs]

    rules = {}
    for rule in spec.get("rules", []):
        key = (rule["from"], rule["to"])
        rules.setdefault(key, set()).update(int(p) for p in rule.get("ports", []))

    return Policy(
        zones=zones,
        rules=rules,
        entry_points=spec.get("entry_points", [INTERNET]),
        crown_jewels=spec.get("crown_jewels", []),
        criticality=spec.get("criticality", {}),
    )


def default_policy(hosts):
    """Synthesise a policy from the hosts themselves — one zone per /24.

    Intra-zone lateral movement is allowed on the usual pivot ports. Any host
    marked `role: "gateway"` lets its own zone reach every other zone, which is
    what the previous hardcoded 10.0.0.5 bridge was doing implicitly. The
    internet reaches only zones that contain a gateway or a web-facing service.
    """
    zones = {}
    for host in hosts:
        try:
            net = ipaddress.ip_network(f"{host['ip']}/24", strict=False)
        except ValueError:
            continue
        zones.setdefault(str(net), [])
        if net not in zones[str(net)]:
            zones[str(net)].append(net)

    rules = {}
    for zone in zones:
        rules[(zone, zone)] = set(DEFAULT_LATERAL_PORTS) | set(DEFAULT_INGRESS_PORTS)

    gateway_zones = set()
    for host in hosts:
        if host.get("role") == "gateway":
            try:
                gateway_zones.add(str(ipaddress.ip_network(f"{host['ip']}/24", strict=False)))
            except ValueError:
                continue

    for gz in gateway_zones:
        for zone in zones:
            if zone != gz:
                # A gateway forwards application traffic as well as carrying
                # lateral protocols, so both sets cross the boundary. Omitting
                # the ingress ports here silently strands any host whose only
                # exposure is web — which is most of a modern estate.
                rules[(gz, zone)] = set(DEFAULT_LATERAL_PORTS) | set(DEFAULT_INGRESS_PORTS)

    # The internet can only touch the perimeter: zones holding a gateway, or
    # failing that (no roles declared) the zone with the lowest network address,
    # which is the conventional DMZ placement in these datasets.
    perimeter = gateway_zones or ({sorted(zones)[0]} if zones else set())
    for zone in perimeter:
        rules[(INTERNET, zone)] = set(DEFAULT_INGRESS_PORTS)

    return Policy(zones=zones, rules=rules, entry_points=[INTERNET], crown_jewels=[])


def resolve_policy(hosts, path=None, min_coverage=0.5, verbose=True):
    """Load `path` if it exists and actually covers these hosts, else synthesise.

    A policy written for one estate applied to another assigns no zone to
    anything, every reachability check fails, and the pipeline produces an empty
    graph with no explanation. That is a silent wrong answer, so coverage is
    checked and a mismatched policy is rejected rather than obeyed.
    """
    if path and os.path.exists(path):
        policy = load_policy(path)
        if not hosts:
            return policy
        covered = sum(1 for h in hosts if policy.zone_of(h.get("ip", "")) is not None)
        coverage = covered / len(hosts)
        if coverage >= min_coverage:
            return policy
        if verbose:
            print(f"  note: {path} maps only {covered}/{len(hosts)} hosts to a zone "
                  f"({coverage:.0%}) — falling back to a synthesised /24 policy.")
    return default_policy(hosts)
