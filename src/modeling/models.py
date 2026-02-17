from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

def fit_random_forest(X: pd.DataFrame, y: pd.Series, seed: int = 123) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf

def fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    monotone_on_n: bool = False,
    seed: int = 123,
):
    """
    XGBoost monotone constraint:
      enforce yield is non-decreasing in N (stabilizes EONR).
    This is NOT a full crop-growth curve constraint (plateau/decline),
    but it’s a strong “biological sanity” baseline for transfer tests.
    """
    import xgboost as xgb

    params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    if monotone_on_n:
        # Build monotone constraint string aligned to X columns
        # +1 for n_rate, 0 for others
        mc = []
        for c in X.columns:
            mc.append(1 if c == "n_rate" else 0)
        params["monotone_constraints"] = tuple(mc)

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

def predict_over_n_grid(
    model,
    X_base: pd.DataFrame,
    n_grid: np.ndarray,
    n_col: str = "n_rate",
) -> pd.DataFrame:
    """
    For a fixed field (covariates), vary N and predict yield.
    Returns long df: [row_id, n_rate, yhat]
    """
    out = []
    X = X_base.copy()
    for n in n_grid:
        X[n_col] = n
        yhat = model.predict(X)
        out.append(pd.DataFrame({"n_rate": n, "yhat": yhat}))
    res = pd.concat(out, axis=0, ignore_index=True)
    return res
