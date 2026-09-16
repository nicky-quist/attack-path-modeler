"""The hosted builder's JavaScript pipeline must build the graph Python builds.

GitHub Pages cannot run Python, so builder.html runs assets/pipeline.js instead
of POST /api/generate. These tests feed both implementations the same hosts,
EPSS scores, KEV entries and segmentation policy, and require the same graph,
chain and choke points back. The network lookups are the only part not
compared: they are replaced here by fixed scores on both sides.

Skipped when Node.js is not installed. GitHub's Ubuntu runners include it.
"""
import contextlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from src import exploitability
from src.analysis import find_choke_points, most_probable_path
from src.export import export_graph
from src.graph import build_graph
from src.segmentation import resolve_policy
from src.synthetic import ZONES, _rules, generate_synthetic_network

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")

RUNNER = r"""
const fs = require("fs");
const AttackPath = require(process.argv[1]);
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = cases.map(c => {
  const hosts = AttackPath.annotateHosts(c.hosts, new Map(Object.entries(c.epss)), new Set(c.kev));
  return AttackPath.analyze(hosts, c.policy, "parity").graph;
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def synthetic_case(seed, declare_crown_jewels=True):
    hosts, policy = generate_synthetic_network(seed=seed)
    spec = {
        "zones": {name: [z["cidr"]] for name, z in ZONES.items()},
        "rules": [{"from": s, "to": d, "ports": ports} for s, d, ports in _rules()],
        "entry_points": ["internet"],
        "crown_jewels": policy.crown_jewels if declare_crown_jewels else [],
    }
    cves = [v["cve"] for h in hosts for v in h["vulns"]]
    # Real scores for a few CVEs and KEV for one, so every branch of the
    # exploitability logic is exercised, not only the CVSS fallback.
    epss = {cve: round(0.01 + (i % 7) * 0.13, 4) for i, cve in enumerate(cves) if i % 3 == 0}
    kev = cves[1:2]
    return {"hosts": hosts, "policy": spec, "epss": epss, "kev": kev}


def spec_hosts(entries):
    hosts = []
    for hostname, ip, cvss, port, service, role in entries:
        host = {"ip": ip, "hostname": hostname,
                "vulns": [{"cve": f"CVE-2099-{10000 + i}", "cvss": c, "port": port, "service": service}
                          for i, c in enumerate(cvss)]}
        if role:
            host["role"] = role
        hosts.append(host)
    return hosts


def python_graph(case, workdir):
    hosts = json.loads(json.dumps(case["hosts"]))
    with mock.patch.object(exploitability, "fetch_epss", return_value=dict(case["epss"])), \
         mock.patch.object(exploitability, "load_kev", return_value=set(case["kev"])):
        exploitability.annotate_hosts(hosts)

    policy_path = None
    if case["policy"] is not None:
        policy_path = os.path.join(workdir, "policy.json")
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(case["policy"], f)

    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        policy = resolve_policy(hosts, policy_path)
        G = build_graph(hosts, policy)
        chokes = find_choke_points(G, policy, top=5)
        risk = most_probable_path(G, policy)
        out = os.path.join(workdir, "graph.json")
        export_graph(G, out, risk_path=risk["path"] if risk else [], source_label="parity",
                     path_probability=risk["probability"] if risk else None,
                     choke_points=[{"id": c[0], "share": round(c[1], 4)} for c in chokes])
    with open(out, encoding="utf-8") as f:
        return json.load(f)


@unittest.skipUnless(NODE, "Node.js is not installed")
class BrowserPipelineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "data", "segmentation.json"), encoding="utf-8") as f:
            bundled_policy = json.load(f)

        cls.cases = {
            # Seed 3 has a choke-point share Python rounds down (0.56875 -> 0.5687) and
            # seed 9 has equal-cost chains straddling the top-k cut, so both catch
            # a port that is right in general but differs from Python on ties.
            **{f"synthetic seed {s}": synthetic_case(s) for s in (0, 1, 2, 3, 9)},
            "synthetic, crown jewels inferred": synthetic_case(7, declare_crown_jewels=False),
            "builder example on the bundled policy": {
                "hosts": spec_hosts([("web-server", "10.0.0.10", [10.0, 6.5], "443", "https", "gateway"),
                                     ("file-server", "10.0.1.20", [8.8], "445", "smb", None)]),
                "policy": bundled_policy, "epss": {"CVE-2099-10000": 0.944}, "kev": ["CVE-2099-10000"]},
            "uncovered hosts fall back to a /24 policy": {
                "hosts": spec_hosts([("proxy", "10.10.0.1", [9.8], "443", "https", "gateway"),
                                     ("jenkins", "10.10.0.15", [9.8], "8080", "http", None),
                                     ("moveit", "10.10.0.10", [9.8], "443", "https", None),
                                     ("db", "10.10.1.20", [9.8], "8443", "https", None)]),
                "policy": bundled_policy, "epss": {}, "kev": []},
            "no gateway, no policy": {
                "hosts": spec_hosts([("a", "192.168.5.4", [7.1], "3389", "rdp", None),
                                     ("b", "192.168.5.9", [5.0, 9.1], "22", "ssh", None),
                                     ("c", "192.168.2.1", [8.0], "443", "https", None),
                                     ("d", "192.168.2.7", [6.2], "445", "smb", None),
                                     ("e", "192.168.2.8", [4.4], "", "", None)]),
                "policy": None, "epss": {}, "kev": []},
        }

        cls.tmp = tempfile.mkdtemp()
        cls.expected = {name: python_graph(case, cls.tmp) for name, case in cls.cases.items()}

        names = list(cls.cases)
        cases_path = os.path.join(cls.tmp, "cases.json")
        result_path = os.path.join(cls.tmp, "result.json")
        with open(cases_path, "w", encoding="utf-8") as f:
            json.dump([cls.cases[n] for n in names], f)
        subprocess.run([NODE, "-e", RUNNER, os.path.join(REPO, "assets", "pipeline.js"), cases_path, result_path],
                       check=True, capture_output=True, text=True)
        with open(result_path, encoding="utf-8") as f:
            cls.actual = dict(zip(names, json.load(f)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assertClose(self, a, b, msg):
        if isinstance(a, float) or isinstance(b, float):
            self.assertTrue(a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9), f"{msg}: {a} != {b}")
        else:
            self.assertEqual(a, b, msg)

    def test_every_case_produces_a_non_trivial_graph(self):
        for name, graph in self.expected.items():
            self.assertGreater(len(graph["links"]), 0, name)
        self.assertTrue(any(len(g["riskPath"]) > 2 for g in self.expected.values()),
                        "no case exercises a multi-hop chain")

    def test_same_hosts(self):
        for name in self.cases:
            with self.subTest(name):
                py, js = self.expected[name]["nodes"], self.actual[name]["nodes"]
                self.assertEqual([n["id"] for n in py], [n["id"] for n in js])
                for p, j in zip(py, js):
                    for key in p:
                        if key != "vulns":
                            self.assertClose(p[key], j[key], f"{p['id']}.{key}")
                    self.assertEqual(p["vulns"], j["vulns"], p["id"])

    def test_same_edges(self):
        for name in self.cases:
            with self.subTest(name):
                py = {(l["source"], l["target"]): l for l in self.expected[name]["links"]}
                js = {(l["source"], l["target"]): l for l in self.actual[name]["links"]}
                self.assertEqual(set(py), set(js))
                for edge, link in py.items():
                    for key in link:
                        self.assertClose(link[key], js[edge][key], f"{edge}.{key}")

    def test_same_most_probable_chain(self):
        for name in self.cases:
            with self.subTest(name):
                self.assertEqual(self.expected[name]["riskPath"], self.actual[name]["riskPath"])
                self.assertClose(self.expected[name]["riskPathProbability"],
                                 self.actual[name]["riskPathProbability"], "probability")

    def test_same_choke_points(self):
        # Python counts through a set of IP strings, whose iteration order is
        # hash-randomised, so hosts tied on share can swap places between runs.
        # Compare the shares, and each host that both sides list.
        for name in self.cases:
            with self.subTest(name):
                py = {c["id"]: c["share"] for c in self.expected[name]["chokePoints"]}
                js = {c["id"]: c["share"] for c in self.actual[name]["chokePoints"]}
                self.assertEqual(sorted(py.values()), sorted(js.values()))
                for host in py.keys() & js.keys():
                    self.assertEqual(py[host], js[host], host)


if __name__ == "__main__":
    unittest.main()
