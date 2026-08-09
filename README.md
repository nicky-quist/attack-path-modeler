# Graph-Based Attack Path Modeler

A Python pipeline that ingests Nessus vulnerability scan data into a NetworkX graph model to map host-to-host attack paths based on exposed services and known CVEs. Includes a GNN classifier built with PyTorch Geometric and an interactive D3.js dashboard.

---

## Features

- **Nessus Parser** — ingests `.nessus` XML scan files and extracts hosts, CVEs, CVSS scores, ports, and services
- **Attack Graph** — builds a directed NetworkX graph where edges represent lateral movement paths between hosts, weighted by exploit ease (1/CVSS)
- **Cross-Subnet Modeling** — bridges network segments through gateway nodes to model realistic multi-hop attack chains
- **Path Analysis** — ranks all attack paths by cumulative risk score and identifies choke points using betweenness centrality
- **GNN Classifier** — PyTorch Geometric GCNConv model for edge-level high-risk lateral movement prediction
- **Matplotlib Visualization** — static attack graph with color-coded risk levels and highlighted attack paths
- **D3.js Dashboard** — interactive browser visualization with draggable nodes, hover tooltips showing CVE details, and highlighted attack chains

---

## Project structure
```
attack-path-modeler/
├── data/
│   ├── sample.nessus   # Sample Nessus scan data
│   └── graph.json      # Exported graph for D3 dashboard
├── src/
│   ├── parser.py        # Nessus XML parser
│   ├── graph.py          # NetworkX graph builder
│   ├── analysis.py       # Attack path ranking and choke point detection
│   ├── export.py         # JSON export for D3
│   └── gnn.py             # PyTorch Geometric GNN classifier
├── main.py               # Main pipeline runner
├── dashboard.html        # Interactive D3.js visualization
└── requirements.txt
```

---

## Installation
```bash
pip install -r requirements.txt
```
Requirements: `networkx`, `matplotlib`, `torch`, `torch-geometric`

## Usage
Run the full pipeline:
```bash
python main.py
```
This will:
1. Parse `data/sample.nessus`
2. Build the attack graph
3. Find and display the highest risk attack path
4. Export `data/graph.json`
5. Train the GNN classifier on synthetic data

Launch the interactive dashboard:
```bash
python -m http.server 8080
```
Then open `http://localhost:8080/dashboard.html` in your browser.

## How it works

### 1. Graph construction
Each host becomes a node with attributes: hostname, list of CVEs, and max CVSS score. Directed edges connect hosts on the same subnet, weighted by `1 / max_cvss` — lower weight means easier to exploit. A VPN gateway node bridges the two subnets to model cross-network lateral movement.

### 2. Attack path ranking
Uses Dijkstra's algorithm (`nx.shortest_path`) weighted by exploit ease to find the most dangerous attack chain. All simple paths are enumerated and ranked by cumulative risk score.

### 3. GNN classifier
Node features: `[max_cvss, num_vulns, num_open_ports, is_internal_subnet]`

Two GCNConv layers learn node embeddings from the graph structure. Source and target embeddings are concatenated per edge and fed into a linear classifier that predicts HIGH/LOW risk for each lateral movement path.

> **Note:** GNN accuracy scales with real scan data. With synthetic or small datasets the model defaults to majority-class prediction — this is a data variance issue, not an architecture issue.

### 4. D3.js dashboard
- Node color and size encode CVSS risk level (red = critical, orange = high)
- Hover any node to see its full CVE list
- Hover any edge to see the CVE and service that enables that lateral movement
- Red arrows highlight the highest risk attack chain
- Drag nodes to rearrange the layout

## Sample attack chain
Running against the included sample data produces this cross-subnet attack path:

```
web-server (CVE-2021-44228, CVSS 10.0)
    → vpn-gateway (CVE-2017-7508, CVSS 7.5)
        → domain-controller (CVE-2020-1472 ZeroLogon, CVSS 10.0)
```

An attacker exploits Log4Shell on the public web server, pivots through the VPN gateway, and achieves domain compromise via ZeroLogon.

## Future work
- Train GNN on real Nessus scan data from heterogeneous networks
- Add MITRE ATT&CK technique mapping per CVE
- Implement temporal attack path modeling (time-based exploit chaining)
- Add remediation priority scoring based on choke point centrality
