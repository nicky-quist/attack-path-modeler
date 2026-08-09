"""
Looks up real CVSS scores from the NVD API by CVE ID, so hosts can be defined
by just typing known CVEs instead of needing an actual Nessus scan.
"""
import json
import os
import time
import urllib.error
import urllib.request

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", ".cve_cache.json")


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_cvss(cve_id, cache=None, delay=6):
    """Fetch the CVSS base score + description for a CVE ID from NVD.

    Prefers CVSS v3.1, falls back to v3.0 then v2 if that's all NVD has for
    an older CVE. Results are cached to data/.cve_cache.json so re-running
    the pipeline doesn't re-query NVD for CVEs it's already looked up.

    `delay` is a sleep applied after any real network call (not cache hits)
    to stay under NVD's unauthenticated rate limit of ~5 requests/30s.
    """
    cache = _load_cache() if cache is None else cache
    if cve_id in cache:
        return cache[cve_id]

    req = urllib.request.Request(
        f"{NVD_URL}?cveId={cve_id}",
        headers={"User-Agent": "attack-path-modeler"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise ValueError(f"NVD lookup failed for {cve_id}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"NVD lookup failed for {cve_id}: {e.reason}") from e

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        raise ValueError(f"{cve_id} not found in NVD")

    cve = vulns[0]["cve"]
    metrics = cve.get("metrics", {})

    cvss = 0.0
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            cvss = metrics[key][0]["cvssData"]["baseScore"]
            break

    description = ""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            description = d.get("value", "")
            break

    result = {"cvss": cvss, "description": description}
    cache[cve_id] = result
    _save_cache(cache)
    time.sleep(delay)
    return result


def build_hosts_from_known_cves(spec):
    """Convert a {hostname: {...}} spec into the same `hosts` structure
    parse_nessus() produces, so it drops straight into build_graph().

    spec example:
      {
        "web-server": {
          "ip": "192.168.10.10", "cves": ["CVE-2021-44228"],
          "port": "443", "service": "https"
        },
        "edge-firewall": {
          "ip": "192.168.10.1", "role": "gateway",
          "cves": ["CVE-2023-27997"], "port": "443", "service": "https"
        }
      }

    A host with "role": "gateway" bridges to every host outside its own
    subnet (see build_graph in src/graph.py) — set this on whatever sits
    between network segments (firewall, VPN concentrator, jump host, etc.).
    """
    cache = _load_cache()
    hosts = []
    for hostname, info in spec.items():
        vulns = []
        for cve_id in info.get("cves", []):
            result = fetch_cvss(cve_id, cache=cache)
            vulns.append({
                "cve": cve_id,
                "cvss": result["cvss"],
                "port": info.get("port", ""),
                "service": info.get("service", ""),
            })
        host = {"ip": info["ip"], "hostname": hostname, "vulns": vulns}
        if "role" in info:
            host["role"] = info["role"]
        hosts.append(host)
    return hosts
