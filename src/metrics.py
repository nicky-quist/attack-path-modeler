"""Classification metrics, reported identically for every model."""


def binary_metrics(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def format_table(rows):
    """rows: [(name, metrics_dict)] -> a fixed-width comparison table."""
    header = f"{'model':<34} {'acc':>7} {'prec':>7} {'recall':>7} {'F1':>7}"
    lines = [header, "-" * len(header)]
    for name, m in rows:
        lines.append(f"{name:<34} {m['accuracy']:>7.3f} {m['precision']:>7.3f} "
                     f"{m['recall']:>7.3f} {m['f1']:>7.3f}")
    return "\n".join(lines)
