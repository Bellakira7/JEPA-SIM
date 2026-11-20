"""Scene-cluster confidence intervals for finite per-frame metrics."""
import numpy as np


def cluster_boot(values, groups, repetitions=1000, rng=None):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if values.ndim != 1 or values.shape != groups.shape:
        raise ValueError('One scene identifier is required per metric value')
    if repetitions < 1:
        raise ValueError('Positive bootstrap repetitions required')
    valid = np.isfinite(values)
    values, groups = values[valid], groups[valid]
    if not len(values):
        return (float('nan'),) * 3
    rng = np.random.default_rng(0) if rng is None else rng
    labels = np.unique(groups)
    sums = np.array([values[groups == group].sum() for group in labels])
    counts = np.array([(groups == group).sum() for group in labels])
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(labels), len(labels))
        estimates.append(sums[sample].sum() / counts[sample].sum())
    lower, upper = np.quantile(estimates, [.025, .975])
    return float(values.mean()), float(lower), float(upper)
