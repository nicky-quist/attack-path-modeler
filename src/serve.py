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

from src.analysis import find_choke_points, find_highest_risk_path
from src.cve_lookup import build_hosts_from_known_cves
from src.export import export_graph
from src.graph import build_graph

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            spec = body.get("hosts", {})
            if not spec:
                raise ValueError("No hosts provided.")

            hosts = build_hosts_from_known_cves(spec)
            G = build_graph(hosts)
            choke_points = find_choke_points(G)
            risk = find_highest_risk_path(G)
            risk_path_ips = risk["path"] if risk else []

            export_graph(G, "data/graph.json", risk_path=risk_path_ips,
                         source_label="known CVEs — typed in via web form")

            result = {
                "ok": True,
                "nodeCount": G.number_of_nodes(),
                "edgeCount": G.number_of_edges(),
                "chokePoints": [
                    {"hostname": G.nodes[ip]["hostname"], "score": round(score, 3)}
                    for ip, score in choke_points[:5]
                ],
                "riskPath": [G.nodes[ip]["hostname"] for ip in risk_path_ips],
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
            httpd = socketserver.TCPServer(("", candidate), DashboardRequestHandler)
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
