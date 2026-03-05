import numpy as np


def acceptance_rate(accepted, total):
    """Compute acceptance rate from counts."""
    if total == 0:
        return 0.0
    return accepted / total


def mean_and_var(samples):
    """Return empirical mean and variance of samples."""
    samples = np.asarray(samples)
    return np.mean(samples, axis=0), np.var(samples, axis=0)


def autocorrelation(samples, lag=10):
    """
    Simple autocorrelation estimate for a given lag.
    Works for 1D chains.
    """
    samples = np.asarray(samples)

    if lag >= len(samples):
        raise ValueError("Lag is larger than the chain length.")

    x = samples - np.mean(samples)

    num = np.sum(x[:-lag] * x[lag:])
    den = np.sum(x * x)

    return num / den


def trace_summary(samples):
    """
    Quick summary useful during experiments.
    """
    mean, var = mean_and_var(samples)

    return {
        "mean": mean,
        "variance": var,
        "n_samples": len(samples),
    }