from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

FLOOR = 1e-8

GRID_STEP_CM = 2.0
HEADLINE_SIGMA_CM = 10.0
ALTERNATE_SIGMA_CM = 15.0


def _clean(Y: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(Y, dtype=np.float64), nan=0.0), 0.0, None)


def sum_normalize(Y: np.ndarray) -> np.ndarray:
    Y = _clean(Y)
    return Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-12, None)


def broaden(Y: np.ndarray, sigma_cm: float) -> np.ndarray:
    sigma_points = float(sigma_cm) / GRID_STEP_CM
    smoothed = gaussian_filter1d(_clean(Y), sigma_points, axis=1, mode="constant", cval=0.0)
    return sum_normalize(smoothed)


def per_spectrum_sid(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    a = sum_normalize(Y_true) + FLOOR
    b = sum_normalize(Y_pred) + FLOOR
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    return (a * np.log(a / b) + b * np.log(b / a)).sum(axis=1)


def per_spectrum_sis(
    Y_true: np.ndarray, Y_pred: np.ndarray, sigma_cm: float | None = HEADLINE_SIGMA_CM
) -> np.ndarray:
    if sigma_cm:
        Y_true = broaden(Y_true, sigma_cm)
        Y_pred = broaden(Y_pred, sigma_cm)
    return 1.0 / (1.0 + per_spectrum_sid(Y_true, Y_pred))


def sis(Y_true: np.ndarray, Y_pred: np.ndarray, sigma_cm: float | None = HEADLINE_SIGMA_CM) -> float:
    return float(per_spectrum_sis(Y_true, Y_pred, sigma_cm).mean())


def sid(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(per_spectrum_sid(Y_true, Y_pred).mean())


def cosine(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    a, b = sum_normalize(Y_true), sum_normalize(Y_pred)
    numerator = (a * b).sum(axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((numerator / denominator).mean())


def mae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.abs(sum_normalize(Y_true) - sum_normalize(Y_pred)).mean())


def all_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    return {
        "SID": sid(Y_true, Y_pred),
        "SIS": sis(Y_true, Y_pred, sigma_cm=None),
        f"SIS_b{int(HEADLINE_SIGMA_CM)}": sis(Y_true, Y_pred, HEADLINE_SIGMA_CM),
        f"SIS_b{int(ALTERNATE_SIGMA_CM)}": sis(Y_true, Y_pred, ALTERNATE_SIGMA_CM),
        "cosine": cosine(Y_true, Y_pred),
        "MAE": mae(Y_true, Y_pred),
    }
