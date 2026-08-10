"""
Serves the repo root over HTTP and opens the dashboard in a browser tab,
so viewing results doesn't require a second manual command.
"""
import http.server
import os
import socketserver
import webbrowser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve_dashboard(port=8080, open_browser=True):
    os.chdir(REPO_ROOT)
    httpd = socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler)

    url = f"http://localhost:{port}/dashboard.html"
    print(f"\nServing dashboard at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.shutdown()
