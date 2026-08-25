"""
The static risk report.

This used to be a second force-directed node-link diagram — the same hosts, the
same colour scale and the same highlighted chain the D3 dashboard already draws,
except you could not drag it or hover it. Two renderings of one picture is not
two visualisations.

The dashboard owns topology, because that is the view interaction actually helps
with: dragging nodes apart to see structure, hovering an edge for the CVE behind
it. So this figure answers the questions a node-link graph is bad at, and one of
them is the project's central claim, which until now was asserted in prose and
shown nowhere:

  1. CVSS is severity, not likelihood. Plotted against EPSS, the correlation is
     weak enough to see by eye — including CVEs with identical CVSS whose
     exploitation probabilities differ by an order of magnitude.
  2. Which chains are actually most probable, and by how much. A graph drawing
     can highlight one path; it cannot rank ten.
  3. Which hosts sit on the most chains — the containment decision.
"""
import matplotlib
import matplotlib.pyplot as plt

from src.analysis import find_choke_points, top_attack_paths

BG = "#1a1a2e"
PANEL = "#22223f"
FG = "#e8e8f0"
MUTED = "#9aa0b5"
ACCENT = "#ff6b6b"
KEV_COLOR = "#ff6b6b"
EPSS_COLOR = "#4dabf7"


def _dark(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#3a3a5c")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(alpha=0.15, color=MUTED)
    ax.set_axisbelow(True)


def _collect_vulns(G):
    seen = {}
    for node in G.nodes:
        for vuln in G.nodes[node].get("vulns", []):
            cve = vuln.get("cve")
            if cve and cve not in seen:
                seen[cve] = vuln
    return list(seen.values())


def plot_severity_vs_likelihood(ax, G):
    """The claim the whole project rests on, finally drawn."""
    vulns = _collect_vulns(G)
    if not vulns:
        ax.text(0.5, 0.5, "no CVEs", ha="center", va="center", color=MUTED)
        _dark(ax)
        return

    for kev, color, label in ((True, KEV_COLOR, "on CISA KEV"), (False, EPSS_COLOR, "EPSS only")):
        subset = [v for v in vulns if bool(v.get("in_kev")) is kev]
        if subset:
            ax.scatter([v.get("cvss", 0) for v in subset],
                       [v.get("p_exploit", 0) for v in subset],
                       s=90, alpha=0.85, color=color, edgecolors="#12121f",
                       linewidths=1.2, label=label, zorder=3)

    # Find CVEs that share a CVSS but diverge sharply in probability — the single
    # clearest demonstration that these two numbers measure different things.
    by_cvss = {}
    for v in vulns:
        by_cvss.setdefault(round(float(v.get("cvss", 0)), 1), []).append(v)
    best = None
    for cvss, group in by_cvss.items():
        if len(group) < 2:
            continue
        lo = min(group, key=lambda v: v.get("p_exploit", 0))
        hi = max(group, key=lambda v: v.get("p_exploit", 0))
        gap = hi.get("p_exploit", 0) - lo.get("p_exploit", 0)
        if best is None or gap > best[0]:
            best = (gap, cvss, lo, hi)

    if best and best[0] > 0.3:
        _gap, cvss, lo, hi = best
        ax.annotate("", xy=(cvss, hi["p_exploit"]), xytext=(cvss, lo["p_exploit"]),
                    arrowprops=dict(arrowstyle="<->", color=FG, lw=1.6, alpha=0.9))
        ax.annotate(f"both CVSS {cvss}\n{lo['cve']}  p={lo['p_exploit']:.3f}\n"
                    f"{hi['cve']}  p={hi['p_exploit']:.3f}",
                    xy=(cvss, (lo["p_exploit"] + hi["p_exploit"]) / 2),
                    xytext=(-14, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=FG,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#2d2d52",
                              edgecolor="#4a4a7a", alpha=0.95))

    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("CVSS base score  (severity)")
    ax.set_ylabel("P(exploit)  —  EPSS / KEV  (likelihood)")
    ax.set_title("Severity does not predict likelihood", fontsize=12, pad=10)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(MUTED)
    _dark(ax)


def plot_top_chains(ax, G, policy, top=6):
    """A graph drawing highlights one path. This ranks them."""
    chains = top_attack_paths(G, policy, k=40)[:top]
    if not chains:
        ax.text(0.5, 0.5, "no chains to a crown jewel", ha="center", va="center", color=MUTED)
        _dark(ax)
        return

    chains = list(reversed(chains))
    # Chain names go INSIDE the bar. As y-tick labels they were clipped by the
    # axes margin, and abbreviating the middle to an ellipsis made distinct
    # chains render as identical strings.
    labels = [" → ".join(G.nodes[n]["hostname"] for n in c["path"]) for c in chains]
    probs = [c["probability"] for c in chains]

    colors = [ACCENT if i == len(chains) - 1 else "#5c5c8a" for i in range(len(chains))]
    ax.barh(range(len(chains)), probs, color=colors, alpha=0.9, height=0.7)
    ax.set_yticks([])

    for i, (label, p) in enumerate(zip(labels, probs)):
        inside = p > 0.35
        ax.text(0.012 if inside else p + 0.02, i, label,
                va="center", ha="left", fontsize=8,
                color="#ffffff" if inside else FG,
                fontweight="bold" if i == len(chains) - 1 else "normal")
        ax.text(min(p + 0.015, 1.02), i, f"{p:.3f}", va="center", fontsize=8.5, color=MUTED)

    spread = max(probs) - min(probs)
    subtitle = ""
    if spread < 0.05:
        # Worth saying out loud: when every hop is a KEV entry, the chains are
        # all near-certain and the ranking between them carries no information.
        subtitle = (f"\nall within {spread:.3f} — every hop here is KEV, "
                    "so ranking them says little")

    ax.set_xlim(0, 1.3)
    ax.set_xlabel("P(chain succeeds)  =  ∏ p over its edges")
    ax.set_title(f"Most probable attack chains (top {len(chains)}){subtitle}", fontsize=12, pad=10)
    _dark(ax)


def plot_choke_points(ax, G, policy, top=6):
    """The containment decision: cutting here breaks how many routes?"""
    chokes = find_choke_points(G, policy, top=top)
    if not chokes:
        ax.text(0.5, 0.5, "no shared choke points", ha="center", va="center", color=MUTED)
        _dark(ax)
        return

    chokes = list(reversed(chokes))
    names = [G.nodes[n]["hostname"] for n, _s, _c, _t in chokes]
    shares = [s for _n, s, _c, _t in chokes]
    counts = [(c, t) for _n, _s, c, t in chokes]

    ax.barh(range(len(chokes)), shares, color="#4dabf7", alpha=0.9, height=0.62)
    ax.set_yticks(range(len(chokes)))
    ax.set_yticklabels(names, fontsize=9, color=FG)
    for i, (c, t) in enumerate(counts):
        ax.text(shares[i] + 0.012, i, f"{c}/{t}", va="center", fontsize=8.5, color=MUTED)

    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of the top chains passing through this host")
    ax.set_title("Choke points — contain here to break the most routes", fontsize=12, pad=10)
    _dark(ax)


def risk_report(G, policy, best=None, path="attack_path_report.png", show=True):
    """Three panels the interactive dashboard does not and should not duplicate."""
    if not show:
        matplotlib.use("Agg")

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.18,
                            left=0.07, right=0.97, top=0.87, bottom=0.09)

    plot_severity_vs_likelihood(fig.add_subplot(grid[0, :]), G)
    plot_top_chains(fig.add_subplot(grid[1, 0]), G, policy)
    plot_choke_points(fig.add_subplot(grid[1, 1]), G, policy)

    title = "Attack Path Risk Report"
    if best:
        chain = " → ".join(G.nodes[n]["hostname"] for n in best["path"])
        title += f"\nMost probable chain: {chain}   P = {best['probability']:.4f}"
    fig.suptitle(title, color=FG, fontsize=15, y=0.965)
    fig.text(0.5, 0.018,
             "Topology lives in the interactive dashboard (python main.py --serve). "
             "These are the views a node-link diagram cannot give you.",
             ha="center", color=MUTED, fontsize=9)

    if path:
        fig.savefig(path, dpi=140, facecolor=BG)
        print(f"Risk report saved to {path}")
    if show:
        plt.show()
    plt.close(fig)
