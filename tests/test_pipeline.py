"""Graph construction, exploitability scoring, segmentation, and path analysis."""
import ipaddress
import math
import unittest

from src.analysis import find_choke_points, most_probable_path, top_attack_paths
from src.exploitability import KEV_FLOOR, annotate_hosts, cvss_fallback_probability
from src.graph import build_graph, edge_cost, path_probability
from src.segmentation import INTERNET, Policy, default_policy, resolve_policy
from src.synthetic import generate_synthetic_network


def _hosts():
    return [
        {"ip": "10.0.0.1", "hostname": "web", "vulns": [
            {"cve": "CVE-2099-99001", "cvss": 10.0, "port": "443", "service": "https"}]},
        {"ip": "10.0.0.2", "hostname": "patched", "vulns": []},
        {"ip": "10.0.1.1", "hostname": "db", "vulns": [
            {"cve": "CVE-2099-99002", "cvss": 9.8, "port": "3306", "service": "mysql"}]},
    ]


def _synthetic(seed=0):
    hosts, policy = generate_synthetic_network(seed=seed)
    return build_graph(annotate_hosts(hosts, offline=True), policy), policy


class TestExploitability(unittest.TestCase):
    def test_cvss_fallback_is_monotonic(self):
        scores = [cvss_fallback_probability(c) for c in range(0, 11)]
        self.assertEqual(scores, sorted(scores))

    def test_fallback_stays_in_open_unit_interval(self):
        for c in (0, 5, 10, 99):
            p = cvss_fallback_probability(c)
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_annotate_sets_probability_offline(self):
        """Offline still consults the EPSS/KEV caches on disk, which is correct;
        these CVE IDs cannot appear in either, so the fallback is exercised."""
        for host in annotate_hosts(_hosts(), offline=True):
            for vuln in host["vulns"]:
                self.assertIn("p_exploit", vuln)
                self.assertEqual(vuln["p_source"], "cvss-estimate")
                self.assertGreater(vuln["p_exploit"], 0.0)
                self.assertLess(vuln["p_exploit"], 1.0)

    def test_kev_outranks_a_low_cvss(self):
        """A KEV entry is an observation, not a prediction: it cannot score low
        just because its CVSS is modest."""
        self.assertGreater(KEV_FLOOR, cvss_fallback_probability(5.0))
        self.assertGreaterEqual(max(cvss_fallback_probability(1.0), KEV_FLOOR), KEV_FLOOR)


class TestGraphConstruction(unittest.TestCase):
    def test_no_edge_across_a_boundary_the_policy_forbids(self):
        hosts = annotate_hosts(_hosts(), offline=True)
        policy = Policy(
            zones={"dmz": [ipaddress.ip_network("10.0.0.0/24")],
                   "internal": [ipaddress.ip_network("10.0.1.0/24")]},
            rules={(INTERNET, "dmz"): {443}, ("dmz", "dmz"): {443}},
        )
        G = build_graph(hosts, policy)
        self.assertFalse(G.has_edge("10.0.0.1", "10.0.1.1"),
                         "edge created across a boundary the policy does not permit")

    def test_no_edge_to_a_port_the_target_is_not_vulnerable_on(self):
        """Reachability alone is not enough — there must be something to exploit."""
        hosts = annotate_hosts(_hosts(), offline=True)
        policy = Policy(
            zones={"dmz": [ipaddress.ip_network("10.0.0.0/24")]},
            rules={("dmz", "dmz"): {22}},  # ssh permitted; nothing here is vulnerable on 22
        )
        G = build_graph(hosts, policy)
        self.assertEqual(G.number_of_edges(), 0)

    def test_patched_host_has_no_inbound_edges(self):
        hosts = annotate_hosts(_hosts(), offline=True)
        G = build_graph(hosts, default_policy(hosts))
        self.assertEqual(G.in_degree("10.0.0.2"), 0,
                         "a host with no vulnerabilities was given an attack edge")

    def test_weight_is_negative_log_probability(self):
        self.assertAlmostEqual(edge_cost(0.5), round(-math.log(0.5), 4), places=4)
        G, _ = _synthetic()
        for _u, _v, d in G.edges(data=True):
            self.assertAlmostEqual(d["weight"], round(-math.log(d["p_exploit"]), 4), places=3)

    def test_cheaper_edge_means_more_probable(self):
        self.assertLess(edge_cost(0.9), edge_cost(0.1))

    def test_path_probability_is_the_product_of_edges(self):
        G, policy = _synthetic(seed=1)
        best = most_probable_path(G, policy)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best["probability"],
                               round(path_probability(G, best["path"]), 6), places=5)

    def test_graph_is_not_complete(self):
        """The original model produced a complete digraph. Density 0.5 was the
        symptom that made betweenness centrality meaningless."""
        G, _ = _synthetic()
        n, e = G.number_of_nodes(), G.number_of_edges()
        self.assertLess(e / (n * (n - 1)), 0.35)


class TestSegmentation(unittest.TestCase):
    def test_mismatched_policy_falls_back_instead_of_emptying_the_graph(self):
        hosts = annotate_hosts(_hosts(), offline=True)
        for host in hosts:
            host["ip"] = host["ip"].replace("10.0.", "192.168.")
        policy = resolve_policy(hosts, "data/segmentation.json", verbose=False)
        covered = sum(1 for h in hosts if policy.zone_of(h["ip"]) is not None)
        self.assertEqual(covered, len(hosts))

    def test_matching_policy_is_used_as_written(self):
        hosts = annotate_hosts(_hosts(), offline=True)
        policy = resolve_policy(hosts, "data/segmentation.json", verbose=False)
        self.assertIn("10.0.1.4", policy.crown_jewels)

    def test_zone_lookup(self):
        policy = default_policy(_hosts())
        self.assertIsNotNone(policy.zone_of("10.0.0.1"))
        self.assertIsNone(policy.zone_of("not-an-ip"))

    def test_unparseable_port_is_not_reachable(self):
        policy = default_policy(_hosts())
        zone = policy.zone_of("10.0.0.1")
        self.assertFalse(policy.can_reach(zone, zone, None))
        self.assertFalse(policy.can_reach(zone, zone, "not-a-port"))
        self.assertFalse(policy.can_reach(None, zone, 443))


class TestAnalysis(unittest.TestCase):
    def test_most_probable_path_is_actually_the_best(self):
        G, policy = _synthetic(seed=2)
        best = most_probable_path(G, policy)
        for entry in top_attack_paths(G, policy, k=20):
            self.assertLessEqual(best["cost"] - 1e-9, entry["cost"])

    def test_choke_points_exclude_endpoints(self):
        G, policy = _synthetic()
        names = {n for n, _s, _c, _t in find_choke_points(G, policy, top=10)}
        self.assertNotIn(INTERNET, names)
        for jewel in policy.crown_jewels:
            self.assertNotIn(jewel, names)

    def test_choke_point_share_is_a_fraction(self):
        G, policy = _synthetic()
        for _n, share, count, total in find_choke_points(G, policy, top=10):
            self.assertGreater(share, 0.0)
            self.assertLessEqual(share, 1.0)
            self.assertAlmostEqual(share, count / total, places=6)

    def test_choke_points_are_ranked(self):
        G, policy = _synthetic()
        shares = [s for _n, s, _c, _t in find_choke_points(G, policy, top=10)]
        self.assertEqual(shares, sorted(shares, reverse=True))

    def test_analysis_survives_an_empty_graph(self):
        import networkx as nx
        policy = default_policy([])
        self.assertIsNone(most_probable_path(nx.DiGraph(), policy))
        self.assertEqual(find_choke_points(nx.DiGraph(), policy), [])


if __name__ == "__main__":
    unittest.main()
