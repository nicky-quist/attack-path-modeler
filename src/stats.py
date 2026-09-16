"""
Statistics for comparing models across many independent networks.

Everything here is standard library, so the study's conclusions don't depend
on a statistics package being installed.

Why these, specifically:

  bootstrap_ci     a percentile interval for a mean over networks. Per-network
                   F1 on these graphs is bounded, skewed, and piles up at 1.0,
                   so a normal-theory interval would be wrong in exactly the
                   place it matters.
  paired_summary   every model is trained on the same network and split, so
                   the honest comparison is the per-network *difference*, not
                   two independent means. Two models at 0.90 +/- 0.20 can
                   differ reliably when their per-network gaps are consistent.
  sign_test_p      an exact test on those differences that assumes nothing
                   about their distribution: under "no real difference", a win
                   is a coin flip.
  average_precision
                   threshold-free. F1 at argmax depends on where the decision
                   boundary landed, and class-weighted training moves it on
                   purpose.
  brier, ece       whether the predicted probabilities mean what they say.
"""
import math
import random
import statistics


def mean(values):
    return statistics.fmean(values)


def bootstrap_ci(values, stat=mean, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap: (point estimate, lower, upper)."""
    values = list(values)
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(values) == 1:
        v = stat(values)
        return v, v, v
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int(math.floor((alpha / 2) * n_boot))]
    hi = boots[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return stat(values), lo, hi


def sign_test_p(diffs):
    """Exact two-sided sign test. Ties (zero differences) carry no direction and are dropped."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_summary(a, b, **boot_kwargs):
    """Compare model a against model b on the same networks, pairwise."""
    if len(a) != len(b):
        raise ValueError("paired comparison needs one score per network from each model")
    diffs = [x - y for x, y in zip(a, b)]
    est, lo, hi = bootstrap_ci(diffs, **boot_kwargs)
    return {
        "mean_diff": est,
        "ci_low": lo,
        "ci_high": hi,
        "wins": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
        "sign_test_p": sign_test_p(diffs),
        "n": len(diffs),
    }


def average_precision(scores, labels):
    """Area under the precision-recall curve, stepwise (scikit-learn's definition).

    Tied scores are one threshold, not several: they enter the curve together.
    Returns None when there are no positives, since recall is undefined.
    """
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    total_pos = sum(1 for _, y in pairs if y == 1)
    if total_pos == 0:
        return None

    ap = 0.0
    tp = fp = 0
    prev_recall = 0.0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def brier(probs, labels):
    return statistics.fmean((p - y) ** 2 for p, y in zip(probs, labels))


def ece(probs, labels, n_bins=10):
    """Expected calibration error with equal-width bins.

    For each bin: |observed positive rate - mean predicted probability|,
    weighted by the share of predictions in it.
    """
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    total = len(probs)
    return sum(
        len(b) / total * abs(statistics.fmean(y for _, y in b) - statistics.fmean(p for p, _ in b))
        for b in bins if b
    )
