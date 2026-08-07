import networkx as nx

def build_graph(hosts):
    G = nx.DiGraph()

    # Step 1: Add each host as a node
    for host in hosts:
        G.add_node(host["ip"],
                   hostname=host["hostname"],
                   vulns=host["vulns"],
                   max_cvss=max([v["cvss"] for v in host["vulns"]], default=0))

    for host_a in hosts:
        for host_b in hosts:
            if host_a["ip"] == host_b["ip"]:
                continue
            subnet_a = host_a["ip"].rsplit(".", 1)[0]
            subnet_b = host_b["ip"].rsplit(".", 1)[0]
            if subnet_a == subnet_b:
                max_cvss_b = G.nodes[host_b["ip"]]["max_cvss"]
                if max_cvss_b > 0:
                    worst_vuln = max(host_b["vulns"], key=lambda v: v["cvss"])
                    G.add_edge(host_a["ip"], host_b["ip"],
                                weight=round(1 / max_cvss_b, 4),
                                cve=worst_vuln["cve"],
                                service=worst_vuln["service"],
                                port=worst_vuln["port"])

    # Bridge the two subnets through the vpn-gateway
    vpn_ip = "10.0.0.5"
    internal_hosts = [h for h in hosts if h["ip"].startswith("10.0.1.")]
    for host in internal_hosts:
        max_cvss_b = G.nodes[host["ip"]]["max_cvss"]
        if max_cvss_b > 0:
            worst_vuln = max(host["vulns"], key=lambda v: v["cvss"])
            G.add_edge(vpn_ip, host["ip"],
                       weight=round(1 / max_cvss_b, 4),
                       cve=worst_vuln["cve"],
                       service=worst_vuln["service"],
                       port=worst_vuln["port"])

    return G

if __name__ == "__main__":
    from src.parser import parse_nessus
    hosts = parse_nessus("data/sample.nessus")
    G = build_graph(hosts)
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges(data=True))