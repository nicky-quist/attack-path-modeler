"""
Serves the repo root over HTTP and opens the dashboard in a browser tab,
so viewing results doesn't require a second manual command.
"""
import http.server
import os
import socketserver
import webbrowser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve_dashboard(port=8080, open_browser=True, max_attempts=5):
    os.chdir(REPO_ROOT)

    httpd = None
    for candidate in range(port, port + max_attempts):
        try:
            httpd = socketserver.TCPServer(("", candidate), http.server.SimpleHTTPRequestHandler)
            port = candidate
            break
        except OSError:
            continue  # port already in use — likely a server left running from an earlier session

    if httpd is None:
        print(f"\nCouldn't bind any port from {port} to {port + max_attempts - 1} — "
              f"they're all in use. Close whatever's running on them, or run with --no-serve "
              f"and start `python -m http.server <port>` yourself.")
        return

    url = f"http://localhost:{port}/dashboard.html"
    print(f"\nServing dashboard at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.shutdown()
