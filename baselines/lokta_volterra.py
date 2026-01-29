import numpy as np

from mimic.model_infer import (
    linearize_time_course_16S,
    fit_alpha_Ridge1,
    do_final_fit_Ridge1,
)
from mimic.model_simulate import sim_gLV


def fit_glv_ridge_batch(Y, times, n_a0=3, n_a1=3):
    """
    Fit ridge gLV-L2

    Parameters
    ----------
    Y : np.ndarray, shape (n, T, K)
        Absolute abundances / counts (must be nonnegative).
    times : np.ndarray, shape (T,)
        Time grid (shared across trajectories).
    n_a0, n_a1 : int
        Grid sizes for MIMIC's internal CV over ridge hyperparameters.
    """
    Y = np.asarray(Y, dtype=float)
    times = np.asarray(times, dtype=float)

    if Y.ndim != 3:
        raise ValueError(f"Y must have shape (n, T, K). Got {Y.shape}")
    n, T, K = Y.shape
    if times.shape != (T,):
        raise ValueError(f"times must have shape (T,), got {times.shape}")

    # Build pooled regression problem by stacking per-trajectory linearizations
    X = np.array([], dtype=np.double).reshape(0, K + 1)
    F = np.array([], dtype=np.double).reshape(0, K)

    for i in range(n):
        Xi, Fi = linearize_time_course_16S(Y[i], times)
        X = np.vstack([X, Xi])
        F = np.vstack([F, Fi])

    # Choose ridge hyperparameters via MIMIC (cross-validated grid)
    a0, a1 = fit_alpha_Ridge1(X, F, num_species=K, n_a0=n_a0, n_a1=n_a1)

    mu_hat, M_hat = do_final_fit_Ridge1(X, F, num_species=K, a0=a0, a1=a1)

    predictor = sim_gLV(num_species=K, M=M_hat, mu=mu_hat)
    return predictor, {"mu": mu_hat, "M": M_hat, "a0": a0, "a1": a1}

def forecast_from_prefix(predictor, x_hist, times_full):
    x_hist = np.asarray(x_hist, dtype=float)
    times_full = np.asarray(times_full, dtype=float)
    t0 = x_hist.shape[0]

    x_pred = np.zeros((len(times_full), x_hist.shape[1]), dtype=float)
    x_pred[:t0] = x_hist

    times_future = times_full[t0:] - times_full[t0]  # shift so starts at 0
    y_future, *_ = predictor.simulate(times=times_future, init_species=x_hist[-1])

    x_pred[t0:] = y_future
    return x_pred

def forecast_mean_from_prefix(predictor, x_hist, times_full, clip_nonneg=True):
    """
    Deterministic mean trajectory forecast under gLV ODE.
    """
    x_hist = np.asarray(x_hist, dtype=float)
    times_full = np.asarray(times_full, dtype=float)
    t0, K = x_hist.shape

    x_pred = np.zeros((len(times_full), K), dtype=float)
    x_pred[:t0] = x_hist

    times_future = times_full[t0:] - times_full[t0]  # start at 0
    y_future, *_ = predictor.simulate(times=times_future, init_species=x_hist[-1])

    if clip_nonneg:
        y_future = np.maximum(y_future, 0.0)

    x_pred[t0:] = y_future
    return x_pred