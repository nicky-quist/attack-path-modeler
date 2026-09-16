/*
 * The attack-path pipeline, in the browser.
 *
 * GitHub Pages serves static files only, so the hosted builder cannot reach the
 * Python server behind POST /api/generate. This is a line-for-line port of the
 * parts that request runs (src/cve_lookup.py, src/exploitability.py,
 * src/segmentation.py, src/graph.py, src/analysis.py, src/labels.py and
 * src/export.py), so a graph built here matches the one Python would build.
 * tests/test_browser_pipeline.py holds the two to that on synthetic estates.
 *
 * NVD and FIRST's EPSS API both allow cross-origin requests. CISA's KEV feed on
 * cisa.gov does not, so the catalogue comes from CISA's own mirror of it on
 * GitHub (cisagov/kev-data), which does.
 */
(function (root) {
  "use strict";

  const INTERNET = "internet";
  const DEFAULT_LATERAL_PORTS = [22, 135, 139, 445, 3389, 5985, 5986];
  const DEFAULT_INGRESS_PORTS = [80, 443, 8080, 8443];
  const KEV_FLOOR = 0.9;
  const P_MIN = 1e-4;
  const P_MAX = 0.999;
  const MAX_HOSTS = 50;

  const NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0";
  const EPSS_URL = "https://api.first.org/data/v1/epss";
  const KEV_URL = "https://raw.githubusercontent.com/cisagov/kev-data/develop/known_exploited_vulnerabilities.json";
  const NVD_DELAY_MS = 6000;       // NVD allows ~5 unauthenticated requests per 30 s
  const KEV_MAX_AGE_MS = 24 * 3600 * 1000;

  const CVE_ID_RE = /^CVE-\d{4}-\d{4,}$/i;
  const OCTET = "(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)";
  const IPV4_RE = new RegExp(`^${OCTET}\\.${OCTET}\\.${OCTET}\\.${OCTET}$`);

  const clamp = (p) => Math.max(P_MIN, Math.min(P_MAX, p));
  // Python's round(): rounds the exact binary value, and a true tie goes to the
  // even digit. Math.round(x * 10**n) does neither — 0.56875 is stored as
  // 0.568749999…, which Python rounds to 0.5687 and Math.round to 0.5688.
  // toFixed works from the exact value, so it only needs the tie rule added.
  function round(x, n) {
    if (!Number.isFinite(x)) return x;
    const exact = Math.abs(x).toFixed(n + 25);
    const cut = exact.indexOf(".") + 1 + n;
    if (exact.slice(cut) !== "5" + "0".repeat(24)) return Number(x.toFixed(n));
    const lastDigit = Number(exact[n === 0 ? cut - 2 : cut - 1]);
    const truncated = Number(exact.slice(0, n === 0 ? cut - 1 : cut));
    const magnitude = lastDigit % 2 === 0 ? truncated : truncated + 10 ** -n;
    return Math.sign(x) * Number(magnitude.toFixed(n));
  }

  // ------------------------------------------------------------ IPv4 zones

  function ipToInt(ip) {
    if (!IPV4_RE.test(ip)) return null;
    return ip.split(".").reduce((acc, o) => acc * 256 + Number(o), 0);
  }

  function parseCidr(cidr) {
    const [addr, bits = "32"] = String(cidr).split("/");
    const base = ipToInt(addr);
    const prefix = Number(bits);
    if (base === null || !Number.isInteger(prefix) || prefix < 0 || prefix > 32) {
      throw new Error(`'${cidr}' isn't a valid IPv4 network.`);
    }
    const size = 2 ** (32 - prefix);
    const start = base - (base % size);   // strict=False: host bits are masked off
    return { start, size };
  }

  const inNet = (n, net) => n >= net.start && n < net.start + net.size;
  const slash24 = (ip) => ip.split(".").slice(0, 3).join(".") + ".0/24";

  // ---------------------------------------------------------- segmentation

  class Policy {
    constructor({ zones, rules, entryPoints, crownJewels, criticality }) {
      this.zones = zones;               // Map zone -> [net]
      this.rules = rules;               // Map "from>to" -> Set(ports)
      this.entryPoints = entryPoints && entryPoints.length ? entryPoints : [INTERNET];
      this.crownJewels = crownJewels || [];
      this.criticality = criticality || {};
    }

    zoneOf(ip) {
      const n = ipToInt(ip);
      if (n === null) return null;
      for (const [name, nets] of this.zones) {
        if (nets.some((net) => inNet(n, net))) return name;
      }
      return null;
    }

    criticalityOf(ip) {
      const zone = this.zoneOf(ip);
      if (zone !== null && Object.prototype.hasOwnProperty.call(this.criticality, zone)) {
        return Number(this.criticality[zone]);
      }
      return this.crownJewels.includes(ip) ? 0.9 : 0.5;
    }

    canReach(fromZone, toZone, port) {
      if (fromZone === null || toZone === null) return false;
      if (!/^\s*[+-]?\d+\s*$/.test(String(port ?? ""))) return false;
      const ports = this.rules.get(fromZone + ">" + toZone);
      return Boolean(ports && ports.has(parseInt(port, 10)));
    }
  }

  function loadPolicy(spec) {
    const zones = new Map();
    for (const [name, value] of Object.entries(spec.zones || {})) {
      const cidrs = Array.isArray(value) ? value : (value.cidrs || []);
      zones.set(name, cidrs.map(parseCidr));
    }
    const rules = new Map();
    for (const rule of spec.rules || []) {
      const key = rule.from + ">" + rule.to;
      if (!rules.has(key)) rules.set(key, new Set());
      for (const p of rule.ports || []) rules.get(key).add(parseInt(p, 10));
    }
    return new Policy({
      zones, rules,
      entryPoints: spec.entry_points || [INTERNET],
      crownJewels: spec.crown_jewels || [],
      criticality: spec.criticality || {},
    });
  }

  function defaultPolicy(hosts) {
    const zones = new Map();
    for (const host of hosts) {
      if (ipToInt(host.ip) === null) continue;
      const key = slash24(host.ip);
      if (!zones.has(key)) zones.set(key, [parseCidr(key)]);
    }

    const all = [...DEFAULT_LATERAL_PORTS, ...DEFAULT_INGRESS_PORTS];
    const rules = new Map();
    for (const zone of zones.keys()) rules.set(zone + ">" + zone, new Set(all));

    const gatewayZones = new Set();
    for (const host of hosts) {
      if (host.role === "gateway" && ipToInt(host.ip) !== null) gatewayZones.add(slash24(host.ip));
    }
    for (const gz of gatewayZones) {
      for (const zone of zones.keys()) {
        if (zone !== gz) rules.set(gz + ">" + zone, new Set(all));
      }
    }

    const perimeter = gatewayZones.size ? [...gatewayZones]
      : (zones.size ? [[...zones.keys()].sort()[0]] : []);
    for (const zone of perimeter) rules.set(INTERNET + ">" + zone, new Set(DEFAULT_INGRESS_PORTS));

    return new Policy({ zones, rules, entryPoints: [INTERNET], crownJewels: [] });
  }

  /** Use `spec` if it covers at least half the hosts, else synthesise one zone per /24. */
  function resolvePolicy(hosts, spec, minCoverage = 0.5) {
    if (spec) {
      const policy = loadPolicy(spec);
      if (!hosts.length) return { policy, note: null };
      const covered = hosts.filter((h) => policy.zoneOf(h.ip || "") !== null).length;
      if (covered / hosts.length >= minCoverage) return { policy, note: null };
      return {
        policy: defaultPolicy(hosts),
        note: `The bundled segmentation policy maps only ${covered}/${hosts.length} hosts to a zone, ` +
              "so a policy with one zone per /24 was synthesised instead.",
      };
    }
    return { policy: defaultPolicy(hosts), note: null };
  }

  // -------------------------------------------------------- exploitability

  function cvssFallbackProbability(cvss) {
    return clamp((Number(cvss) / 10) ** 2);
  }

  /** Attach epss, in_kev, p_exploit and p_source to every vuln. Mutates and returns hosts. */
  function annotateHosts(hosts, epss, kev) {
    for (const host of hosts) {
      for (const vuln of host.vulns) {
        const cve = vuln.cve;
        const score = cve && epss.has(cve) ? epss.get(cve) : null;
        const inKev = Boolean(cve) && kev.has(cve);
        let p, source;
        if (score !== null && score !== undefined) { p = Number(score); source = "epss"; }
        else { p = cvssFallbackProbability(vuln.cvss || 0); source = "cvss-estimate"; }
        if (inKev) { p = Math.max(p, KEV_FLOOR); source = "kev"; }
        vuln.epss = score;
        vuln.in_kev = inKev;
        vuln.p_exploit = clamp(p);
        vuln.p_source = source;
      }
    }
    return hosts;
  }

  // ----------------------------------------------------------------- graph

  class DiGraph {
    constructor() { this.nodes = new Map(); this.succ = new Map(); this.pred = new Map(); }
    addNode(id, attrs) { this.nodes.set(id, attrs); this.succ.set(id, new Map()); this.pred.set(id, new Map()); }
    addEdge(u, v, attrs) { this.succ.get(u).set(v, attrs); this.pred.get(v).set(u, attrs); }
    has(id) { return this.nodes.has(id); }
    edge(u, v) { return this.succ.get(u).get(v); }
    inDegree(id) { return this.pred.get(id).size; }
    outDegree(id) { return this.succ.get(id).size; }
    get order() { return this.nodes.size; }
    get size() { let n = 0; for (const s of this.succ.values()) n += s.size; return n; }
    removeNode(id) {
      for (const v of this.succ.get(id).keys()) this.pred.get(v).delete(id);
      for (const u of this.pred.get(id).keys()) this.succ.get(u).delete(id);
      this.nodes.delete(id); this.succ.delete(id); this.pred.delete(id);
    }
  }

  function pExploit(vuln) {
    const p = vuln.p_exploit ?? cvssFallbackProbability(vuln.cvss || 0);
    return clamp(Number(p));
  }

  const edgeCost = (p) => round(-Math.log(p), 4);

  function buildGraph(hosts, policy, includeInternet = true) {
    const G = new DiGraph();
    for (const host of hosts) {
      const vulns = host.vulns;
      const attrs = {
        hostname: host.hostname,
        vulns,
        zone: policy.zoneOf(host.ip),
        max_cvss: vulns.length ? Math.max(...vulns.map((v) => v.cvss)) : 0,
        max_p_exploit: vulns.length ? Math.max(...vulns.map(pExploit)) : 0,
        in_kev: vulns.some((v) => v.in_kev),
        is_crown_jewel: policy.crownJewels.includes(host.ip),
        criticality: "criticality" in host ? Number(host.criticality) : policy.criticalityOf(host.ip),
      };
      if (host.role) attrs.role = host.role;
      G.addNode(host.ip, attrs);
    }
    if (includeInternet) {
      G.addNode(INTERNET, {
        hostname: "internet", vulns: [], zone: INTERNET, max_cvss: 0, max_p_exploit: 0,
        in_kev: false, is_crown_jewel: false, criticality: 0,
      });
    }

    const ids = [...G.nodes.keys()];
    for (const src of ids) {
      const srcZone = G.nodes.get(src).zone;
      for (const tgt of ids) {
        if (src === tgt || tgt === INTERNET) continue;
        let best = null;
        for (const vuln of G.nodes.get(tgt).vulns) {
          if (!policy.canReach(srcZone, G.nodes.get(tgt).zone, vuln.port)) continue;
          const p = pExploit(vuln);
          if (best === null || p > best[0]) best = [p, vuln];
        }
        if (best === null) continue;
        const [p, vuln] = best;
        G.addEdge(src, tgt, {
          weight: edgeCost(p),
          p_exploit: round(p, 6),
          cve: vuln.cve ?? null,
          service: vuln.service ?? null,
          port: vuln.port ?? null,
          in_kev: Boolean(vuln.in_kev),
          p_source: vuln.p_source || "cvss-estimate",
        });
      }
    }

    if (includeInternet && G.outDegree(INTERNET) === 0) G.removeNode(INTERNET);
    return G;
  }

  function pathProbability(G, path) {
    let p = 1;
    for (let i = 1; i < path.length; i++) p *= G.edge(path[i - 1], path[i]).p_exploit;
    return p;
  }

  const pathCost = (G, path) => {
    let c = 0;
    for (let i = 1; i < path.length; i++) c += G.edge(path[i - 1], path[i]).weight;
    return c;
  };

  // ------------------------------------------------------------ shortest paths

  // A binary heap ordered by (distance, insertion count), which is how networkx
  // breaks ties, so equal-cost routes resolve to the same path in both.
  class Heap {
    constructor() { this.a = []; this.n = 0; }
    get length() { return this.a.length; }
    less(i, j) { const x = this.a[i], y = this.a[j]; return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
    push(dist, value, order = this.n++) {
      const a = this.a; a.push([dist, order, value]);
      let i = a.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (!this.less(i, p)) break; [a[i], a[p]] = [a[p], a[i]]; i = p; }
    }
    pop() {
      const a = this.a; const top = a[0]; const last = a.pop();
      if (a.length) {
        a[0] = last; let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < a.length && this.less(l, m)) m = l;
          if (r < a.length && this.less(r, m)) m = r;
          if (m === i) break;
          [a[i], a[m]] = [a[m], a[i]]; i = m;
        }
      }
      return top;
    }
  }

  function dijkstra(G, source, { target = null, ignoreNodes = null, ignoreEdges = null } = {}) {
    const dist = new Map(), paths = new Map([[source, [source]]]), seen = new Map([[source, 0]]);
    const heap = new Heap();
    heap.push(0, source);
    while (heap.length) {
      const [d, , v] = heap.pop();
      if (dist.has(v)) continue;
      dist.set(v, d);
      if (v === target) break;
      for (const [u, e] of G.succ.get(v)) {
        if (ignoreNodes && ignoreNodes.has(u)) continue;
        if (ignoreEdges && ignoreEdges.has(v + ">" + u)) continue;
        const vu = d + e.weight;
        if (dist.has(u)) continue;
        if (!seen.has(u) || vu < seen.get(u)) {
          seen.set(u, vu);
          heap.push(vu, u);
          paths.set(u, [...paths.get(v), u]);
        }
      }
    }
    return { dist, paths };
  }

  // networkx's _bidirectional_dijkstra, step for step. Paths that tie on cost
  // are common (hosts sharing a CVSS score), and which of them fall inside the
  // top k decides the choke-point shares, so the search order has to match
  // Python's exactly, not merely find an equally short path.
  function bidirectionalDijkstra(G, source, target, ignoreNodes, ignoreEdges) {
    if (ignoreNodes.has(source) || ignoreNodes.has(target)) return null;
    if (source === target) return [0, [source]];

    const succ = (v) => [...G.succ.get(v)].filter(([w]) => !ignoreNodes.has(w) && !ignoreEdges.has(v + ">" + w));
    const pred = (v) => [...G.pred.get(v)].filter(([w]) => !ignoreNodes.has(w) && !ignoreEdges.has(w + ">" + v));
    const neighs = [succ, pred];

    const dists = [new Map(), new Map()];
    const paths = [new Map([[source, [source]]]), new Map([[target, [target]]])];
    const seen = [new Map([[source, 0]]), new Map([[target, 0]])];
    const fringe = [new Heap(), new Heap()];
    let counter = 0;   // one tie-break counter shared by both directions, as in networkx
    const push = (dir, d, v) => fringe[dir].push(d, v, counter++);
    push(0, 0, source);
    push(1, 0, target);

    let finalDist = null, finalPath = [];
    let dir = 1;
    while (fringe[0].length && fringe[1].length) {
      dir = 1 - dir;
      const [dist, , v] = fringe[dir].pop();
      if (dists[dir].has(v)) continue;
      dists[dir].set(v, dist);
      if (dists[1 - dir].has(v)) return [finalDist, finalPath];

      for (const [w, e] of neighs[dir](v)) {
        const vwLength = dists[dir].get(v) + e.weight;
        if (dists[dir].has(w)) continue;
        if (!seen[dir].has(w) || vwLength < seen[dir].get(w)) {
          seen[dir].set(w, vwLength);
          push(dir, vwLength, w);
          paths[dir].set(w, [...paths[dir].get(v), w]);
          if (seen[0].has(w) && seen[1].has(w)) {
            const total = seen[0].get(w) + seen[1].get(w);
            if (!finalPath.length || finalDist > total) {
              finalDist = total;
              finalPath = [...paths[0].get(w), ...[...paths[1].get(w)].reverse().slice(1)];
            }
          }
        }
      }
    }
    return null;
  }

  /** networkx.shortest_simple_paths: loopless paths in order of increasing cost. */
  function* shortestSimplePaths(G, source, target) {
    if (!G.has(source) || !G.has(target)) return;
    const listA = [];
    const buffer = new Heap();
    const buffered = new Set();
    const bufferPush = (cost, path) => {
      const key = path.join(" ");
      if (!buffered.has(key)) { buffer.push(cost, path); buffered.add(key); }
    };

    let prev = null;
    for (;;) {
      if (!prev) {
        const found = bidirectionalDijkstra(G, source, target, new Set(), new Set());
        if (!found) return;
        bufferPush(found[0], found[1]);
      } else {
        const ignoreNodes = new Set(), ignoreEdges = new Set();
        for (let i = 1; i < prev.length; i++) {
          const rootPath = prev.slice(0, i);
          const rootLength = pathCost(G, rootPath);
          for (const p of listA) {
            if (rootPath.every((n, j) => p[j] === n) && p.length > i) ignoreEdges.add(p[i - 1] + ">" + p[i]);
          }
          const found = bidirectionalDijkstra(G, rootPath[i - 1], target, ignoreNodes, ignoreEdges);
          if (found) bufferPush(rootLength + found[0], [...rootPath.slice(0, -1), ...found[1]]);
          ignoreNodes.add(rootPath[i - 1]);
        }
      }
      if (!buffer.length) return;
      const path = buffer.pop()[2];
      buffered.delete(path.join(" "));
      yield path;
      listA.push(path);
      prev = path;
    }
  }

  function bfsHops(G, source) {
    const hops = new Map([[source, 0]]);
    let frontier = [source];
    while (frontier.length) {
      const next = [];
      for (const v of frontier) {
        for (const u of G.succ.get(v).keys()) {
          if (!hops.has(u)) { hops.set(u, hops.get(v) + 1); next.push(u); }
        }
      }
      frontier = next;
    }
    return hops;
  }

  // ------------------------------------------------------------- analysis

  function inferCrownJewels(G, policy) {
    let sources = policy.entryPoints.filter((s) => G.has(s));
    if (!sources.length) sources = [...G.nodes.keys()].filter((n) => n !== INTERNET).slice(0, 1);
    if (!sources.length) return [];
    const count = Math.max(1, Math.min(5, Math.floor((G.order - 1) / 10)));

    const depth = new Map();
    for (const src of sources) {
      for (const [node, h] of bfsHops(G, src)) {
        if (node === INTERNET || sources.includes(node)) continue;
        depth.set(node, Math.max(depth.get(node) || 0, h));
      }
    }
    const ranked = [...depth.entries()].sort((a, b) =>
      (b[1] - a[1]) || ((G.nodes.get(b[0]).criticality ?? 0.5) - (G.nodes.get(a[0]).criticality ?? 0.5)));
    return ranked.slice(0, count).map(([n]) => n);
  }

  function sourcesAndTargets(G, policy) {
    if (!G.order) return [[], []];
    let sources = policy.entryPoints.filter((s) => G.has(s));
    if (!sources.length) {
      const roots = [...G.nodes.keys()].filter((n) => G.inDegree(n) === 0);
      sources = roots.length ? roots : [G.nodes.keys().next().value];
    }
    let targets = policy.crownJewels.filter((t) => G.has(t));
    if (!targets.length) targets = inferCrownJewels(G, policy);
    return [sources, targets];
  }

  function topAttackPaths(G, policy, k = 40, maxHops = 6) {
    const [sources, targets] = sourcesAndTargets(G, policy);
    const paths = [];
    for (const src of sources) {
      for (const tgt of targets) {
        if (src === tgt || !G.has(tgt)) continue;
        let i = 0;
        for (const path of shortestSimplePaths(G, src, tgt)) {
          if (i >= k || path.length - 1 > maxHops) break;
          paths.push({ path, cost: round(pathCost(G, path), 4),
                       probability: round(pathProbability(G, path), 6), hops: path.length - 1 });
          i++;
        }
      }
    }
    return paths.sort((a, b) => a.cost - b.cost);
  }

  function findChokePoints(G, policy, top = 10, k = 40) {
    const [sources, targets] = sourcesAndTargets(G, policy);
    const endpoints = new Set([...sources, ...targets, INTERNET]);
    const paths = topAttackPaths(G, policy, k);
    if (!paths.length) return [];
    const counts = new Map();
    for (const entry of paths) {
      for (const node of new Set(entry.path)) {
        if (!endpoints.has(node)) counts.set(node, (counts.get(node) || 0) + 1);
      }
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, top)
      .map(([id, c]) => ({ id, share: c / paths.length, count: c, total: paths.length }));
  }

  function mostProbablePath(G, policy) {
    const [sources, targets] = sourcesAndTargets(G, policy);
    if (!sources.length || !targets.length) return null;
    let best = null;
    for (const src of sources) {
      if (!G.has(src)) continue;
      const { dist, paths } = dijkstra(G, src);
      for (const tgt of targets) {
        if (!dist.has(tgt) || tgt === src) continue;
        const cand = { path: paths.get(tgt), cost: round(dist.get(tgt), 4),
                       probability: round(pathProbability(G, paths.get(tgt)), 6),
                       hops: paths.get(tgt).length - 1 };
        if (best === null || cand.cost < best.cost) best = cand;
      }
    }
    return best;
  }

  // --------------------------------------------------------------- export

  function exportGraph(G, { riskPath = [], sourceLabel = "", pathProbability = null, chokePoints = [] } = {}) {
    const nodes = [...G.nodes.entries()].map(([id, n]) => ({
      id,
      hostname: n.hostname,
      max_cvss: n.max_cvss,
      zone: n.zone ?? null,
      max_p_exploit: round(n.max_p_exploit || 0, 4),
      in_kev: Boolean(n.in_kev),
      criticality: round(n.criticality ?? 0.5, 2),
      is_crown_jewel: Boolean(n.is_crown_jewel),
      vulns: (n.vulns || []).map((v) => ({
        cve: v.cve ?? null, cvss: v.cvss ?? null, service: v.service ?? null, port: v.port ?? null,
        p_exploit: v.p_exploit ? round(v.p_exploit, 4) : null, in_kev: Boolean(v.in_kev),
      })),
    }));
    const links = [];
    for (const [source, out] of G.succ) {
      for (const [target, e] of out) {
        links.push({ source, target, weight: e.weight, p_exploit: e.p_exploit ?? null, in_kev: Boolean(e.in_kev),
                     p_source: e.p_source ?? null, cve: e.cve, service: e.service, port: e.port });
      }
    }
    return {
      nodes, links,
      riskPath: riskPath || [],
      riskPathProbability: pathProbability,
      chokePoints: chokePoints || [],
      sourceLabel: sourceLabel || "",
      generatedAt: new Date().toISOString().replace(/\.\d{3}Z$/, "+00:00"),
    };
  }

  /** Everything after the lookups: policy, graph, ranking, export. No network. */
  function analyze(hosts, policySpec, sourceLabel) {
    const { policy, note } = resolvePolicy(hosts, policySpec);
    const G = buildGraph(hosts, policy);
    const chokes = findChokePoints(G, policy, 5);
    const risk = mostProbablePath(G, policy);
    const graph = exportGraph(G, {
      riskPath: risk ? risk.path : [],
      sourceLabel,
      pathProbability: risk ? risk.probability : null,
      chokePoints: chokes.map((c) => ({ id: c.id, share: round(c.share, 4) })),
    });
    return {
      graph, note,
      nodeCount: G.order,
      edgeCount: G.size,
      chokePoints: chokes.map((c) => ({ hostname: G.nodes.get(c.id).hostname, share: round(c.share, 4) })),
      riskPath: risk ? risk.path.map((id) => G.nodes.get(id).hostname) : [],
      riskPathProbability: risk ? risk.probability : null,
    };
  }

  // --------------------------------------------------------------- lookups

  const store = {
    get(key) { try { return JSON.parse(root.localStorage.getItem(key)); } catch { return null; } },
    set(key, value) { try { root.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage off */ } },
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function validateSpec(spec) {
    const names = Object.keys(spec);
    if (!names.length) throw new Error("No hosts provided.");
    if (names.length > MAX_HOSTS) throw new Error(`Too many hosts (${names.length}) — max ${MAX_HOSTS} per build.`);
    for (const [hostname, info] of Object.entries(spec)) {
      if (!info.ip) throw new Error(`${hostname}: missing an IP address.`);
      if (!IPV4_RE.test(info.ip)) throw new Error(`${hostname}: '${info.ip}' isn't a valid IPv4 address.`);
      if (!info.cves || !info.cves.length) throw new Error(`${hostname}: needs at least one CVE ID.`);
      for (const cve of info.cves) {
        if (!CVE_ID_RE.test(cve)) throw new Error(`'${cve}' doesn't look like a CVE ID (expected e.g. CVE-2021-44228).`);
      }
    }
  }

  async function fetchCvss(cveId, onProgress) {
    const key = "apm:nvd:" + cveId;
    const cached = store.get(key);
    if (cached) return { ...cached, cached: true };

    for (let attempt = 0; ; attempt++) {
      let res;
      try {
        res = await fetch(`${NVD_URL}?cveId=${encodeURIComponent(cveId)}`);
      } catch (e) {
        throw new Error(`NVD lookup failed for ${cveId}: the request was blocked or the network is down.`);
      }
      // NVD answers a burst of unauthenticated requests with 403 or 429; wait and retry.
      if ((res.status === 403 || res.status === 429 || res.status === 503) && attempt < 2) {
        if (onProgress) onProgress(`NVD is rate-limiting — waiting before retrying ${cveId}…`);
        await sleep(NVD_DELAY_MS * (attempt + 1));
        continue;
      }
      if (!res.ok) throw new Error(`NVD lookup failed for ${cveId}: HTTP ${res.status}`);
      const data = await res.json();
      const vulns = data.vulnerabilities || [];
      if (!vulns.length) throw new Error(`${cveId} not found in NVD`);
      const cve = vulns[0].cve, metrics = cve.metrics || {};
      let cvss = 0;
      for (const k of ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]) {
        if (metrics[k] && metrics[k].length) { cvss = metrics[k][0].cvssData.baseScore; break; }
      }
      const description = ((cve.descriptions || []).find((d) => d.lang === "en") || {}).value || "";
      const result = { cvss, description };
      store.set(key, result);
      return { ...result, cached: false };
    }
  }

  async function fetchEpss(cveIds) {
    const cache = store.get("apm:epss") || {};
    const wanted = [...new Set(cveIds)].filter((c) => c && !(c in cache));
    let ok = true;
    for (let i = 0; i < wanted.length; i += 100) {
      const batch = wanted.slice(i, i + 100);
      try {
        const res = await fetch(`${EPSS_URL}?cve=${batch.join(",")}`);
        if (!res.ok) throw new Error(String(res.status));
        const payload = await res.json();
        for (const row of payload.data || []) cache[row.cve] = Number(row.epss);
        for (const cve of batch) if (!(cve in cache)) cache[cve] = null;
      } catch {
        ok = false;
        break;   // offline or rate-limited: the rest fall back to the CVSS estimate
      }
    }
    store.set("apm:epss", cache);
    return { scores: new Map(Object.entries(cache).filter(([, v]) => v !== null)), ok };
  }

  async function loadKev() {
    const cached = store.get("apm:kev");
    if (cached && Date.now() - cached.fetchedAt < KEV_MAX_AGE_MS) return { ids: new Set(cached.ids), ok: true };
    try {
      const res = await fetch(KEV_URL);
      if (!res.ok) throw new Error(String(res.status));
      const payload = await res.json();
      const ids = [...new Set((payload.vulnerabilities || []).map((v) => v.cveID))].sort();
      store.set("apm:kev", { fetchedAt: Date.now(), ids });
      return { ids: new Set(ids), ok: true };
    } catch {
      return { ids: new Set(cached ? cached.ids : []), ok: Boolean(cached) };
    }
  }

  /**
   * The browser equivalent of POST /api/generate: look up every CVE, score
   * exploitability, apply the policy, rank the chains. `spec` is the same
   * {hostname: {ip, cves, port, service, role}} shape the server accepts.
   */
  async function generate(spec, { policySpec = null, sourceLabel = "", onProgress = null } = {}) {
    validateSpec(spec);
    const say = (msg) => { if (onProgress) onProgress(msg); };

    const all = Object.values(spec).flatMap((h) => h.cves);
    const unique = [...new Set(all)];
    const cvss = new Map();
    let networkCalls = 0;
    for (const [i, cveId] of unique.entries()) {
      say(`Looking up ${cveId} in NVD (${i + 1} of ${unique.length})…`);
      if (networkCalls > 0 && !store.get("apm:nvd:" + cveId)) await sleep(NVD_DELAY_MS);
      const result = await fetchCvss(cveId, say);
      if (!result.cached) networkCalls++;
      cvss.set(cveId, result.cvss);
    }

    const hosts = Object.entries(spec).map(([hostname, info]) => {
      const host = {
        ip: info.ip, hostname,
        vulns: info.cves.map((cve) => ({ cve, cvss: cvss.get(cve), port: info.port || "", service: info.service || "" })),
      };
      if (info.role) host.role = info.role;
      return host;
    });

    say("Scoring exploitability with EPSS and CISA KEV…");
    const [epss, kev] = await Promise.all([fetchEpss(all), loadKev()]);
    annotateHosts(hosts, epss.scores, kev.ids);

    const result = analyze(hosts, policySpec, sourceLabel);
    result.warnings = [];
    if (!epss.ok) result.warnings.push("EPSS was unreachable, so some CVEs use the CVSS-derived estimate.");
    if (!kev.ok) result.warnings.push("The CISA KEV catalogue was unreachable, so no CVE is marked as known-exploited.");
    if (result.note) result.warnings.push(result.note);
    return result;
  }

  const api = {
    INTERNET, KEV_FLOOR, generate, analyze, annotateHosts, validateSpec,
    loadPolicy, defaultPolicy, resolvePolicy, buildGraph, mostProbablePath,
    findChokePoints, topAttackPaths, inferCrownJewels, cvssFallbackProbability,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AttackPath = api;
})(typeof window !== "undefined" ? window : globalThis);
