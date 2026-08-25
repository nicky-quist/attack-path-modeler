"""
Serves the repo root over HTTP, opens the dashboard in a browser tab, and
handles POST /api/generate so builder.html can regenerate the graph from
typed-in CVEs without a separate terminal command.
"""
import http.server
import json
import os
import socketserver
import webbrowser

from src.analysis import find_choke_points, most_probable_path
from src.cve_lookup import build_hosts_from_known_cves
from src.export import export_graph
from src.exploitability import annotate_hosts
from src.graph import build_graph
from src.segmentation import resolve_policy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_BODY_BYTES = 100_000  # generous for a host/CVE list; blocks trivial memory-exhaustion attempts
MAX_HOSTS = 50            # keeps k-shortest-path enumeration bounded on user-supplied specs


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Browsers request /favicon.ico unprompted. There isn't one, and the
        # resulting 404 in the log looks like the page failed to load something
        # it needed. Answer it with "no content" instead.
        if self.path in ("/favicon.ico", "/favicon.png"):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return

        # Defense-in-depth against cross-site POSTs: browsers already block this via CORS
        # preflight for a JSON content-type from another origin, but don't rely on that alone.
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_address[1]}",
                                      f"http://localhost:{self.server.server_address[1]}"):
            self._send_json(403, {"ok": False, "error": "Cross-origin request rejected."})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                # Reject without reading the oversized body into memory — but since we're
                # not draining it, the connection can't be reused for a next request.
                self.close_connection = True
                self._send_json(413, {"ok": False, "error": "Request body too large."})
                return

            body = json.loads(self.rfile.read(length))
            spec = body.get("hosts", {})
            if not spec:
                raise ValueError("No hosts provided.")
            if len(spec) > MAX_HOSTS:
                raise ValueError(f"Too many hosts ({len(spec)}) — max {MAX_HOSTS} per request.")

            hosts = build_hosts_from_known_cves(spec)
            hosts = annotate_hosts(hosts)
            policy = resolve_policy(hosts, "data/segmentation.json")
            G = build_graph(hosts, policy)
            choke_points = find_choke_points(G, policy, top=5)
            risk = most_probable_path(G, policy)
            risk_path_ips = risk["path"] if risk else []

            export_graph(G, "data/graph.json", risk_path=risk_path_ips,
                         source_label="known CVEs — typed in via web form",
                         path_probability=risk["probability"] if risk else None,
                         choke_points=[{"id": c[0], "share": round(c[1], 4)} for c in choke_points])

            result = {
                "ok": True,
                "nodeCount": G.number_of_nodes(),
                "edgeCount": G.number_of_edges(),
                "chokePoints": [
                    {"hostname": G.nodes[ip]["hostname"], "share": round(share, 4)}
                    for ip, share, _c, _t in choke_points
                ],
                "riskPath": [G.nodes[ip]["hostname"] for ip in risk_path_ips],
                "riskPathProbability": risk["probability"] if risk else None,
            }
            self._send_json(200, result)

        except Exception as e:
            self._send_json(400, {"ok": False, "error": str(e)})

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quiet down noisy 404s from unrelated browser extensions probing localhost
        if "404" in (args[1] if len(args) > 1 else ""):
            return
        super().log_message(fmt, *args)


def serve_dashboard(port=8080, open_browser=True, max_attempts=5, page="dashboard.html"):
    os.chdir(REPO_ROOT)

    httpd = None
    for candidate in range(port, port + max_attempts):
        try:
            # Bind to loopback only — "" binds all interfaces, which would expose this
            # (unauthenticated, file-writing) server to anyone else on the same network.
            httpd = socketserver.TCPServer(("127.0.0.1", candidate), DashboardRequestHandler)
            port = candidate
            break
        except OSError:
            continue  # port already in use — likely a server left running from an earlier session

    if httpd is None:
        print(f"\nCouldn't bind any port from {port} to {port + max_attempts - 1} — "
              f"they're all in use. Close whatever's running on them, or run with --no-serve "
              f"and start `python -m http.server <port>` yourself.")
        return

    url = f"http://localhost:{port}/{page}"
    print(f"\nServing dashboard at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.shutdown()
