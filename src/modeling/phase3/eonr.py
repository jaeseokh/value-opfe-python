from __future__ import annotations

import numpy as np
import pandas as pd


def profit(n: np.ndarray, y: np.ndarray, p_y: float, p_n: float) -> np.ndarray:
    return p_y * y - p_n * n


def eonr_from_curve(n_grid: np.ndarray, y_grid: np.ndarray, p_y: float, p_n: float) -> float:
    pi = profit(n_grid, y_grid, p_y, p_n)
    return float(n_grid[int(np.nanargmax(pi))])


def fit_true_field_gam_and_eonr(
    df_field: pd.DataFrame,
    y_col: str,
    n_col: str,
    n_grid: np.ndarray,
    p_y: float,
    p_n: float,
) -> tuple[float, np.ndarray]:
    """
    True EONR from within-field GAM.
    Returns (nan, nan-grid) if GAM cannot be fit.
    """
    d = df_field[[y_col, n_col]].dropna().copy()
    if d.empty or d[n_col].nunique() < 4 or len(d) < 8:
        return float("nan"), np.full_like(n_grid, np.nan, dtype=float)

    X = d[[n_col]].to_numpy()
    y = d[y_col].to_numpy()

    try:
        from pygam import LinearGAM, s

        gam = LinearGAM(s(0, n_splines=12)).fit(X, y)
        y_grid = gam.predict(n_grid.reshape(-1, 1))
        eonr = eonr_from_curve(n_grid, y_grid, p_y, p_n)
        return eonr, y_grid
    except Exception:
        return float("nan"), np.full_like(n_grid, np.nan, dtype=float)

