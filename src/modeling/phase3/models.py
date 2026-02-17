from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def fit_random_forest(X: pd.DataFrame, y: pd.Series, seed: int = 123) -> Pipeline:
    """
    RF with median imputation for missing soil/topo/weather values.
    """
    n_estimators = int(os.getenv("PHASE3_RF_N_ESTIMATORS", "600"))
    min_samples_leaf = int(os.getenv("PHASE3_RF_MIN_SAMPLES_LEAF", "5"))
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    monotone_on_n: bool = False,
    seed: int = 123,
):
    """
    XGBoost baseline with optional monotone constraint on n_rate.
    """
    import xgboost as xgb

    X_num = X.copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    n_estimators = int(os.getenv("PHASE3_XGB_N_ESTIMATORS", "1200"))
    learning_rate = float(os.getenv("PHASE3_XGB_LEARNING_RATE", "0.03"))
    max_depth = int(os.getenv("PHASE3_XGB_MAX_DEPTH", "6"))
    params = dict(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    if monotone_on_n:
        mc = tuple(1 if c == "n_rate" else 0 for c in X_num.columns)
        params["monotone_constraints"] = mc

    model = xgb.XGBRegressor(**params)
    model.fit(X_num, y)
    return model


def predict_over_n_grid(
    model,
    X_base: pd.DataFrame,
    n_grid: np.ndarray,
    n_col: str = "n_rate",
) -> pd.DataFrame:
    def _refresh_interactions(x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        # Generic refresh for explicitly engineered interaction columns n_x_<var>.
        for c in out.columns:
            if not c.startswith("n_x_"):
                continue
            var = c[len("n_x_") :]
            if (n_col in out.columns) and (var in out.columns):
                out[c] = out[n_col] * out[var]
        return out

    out = []
    X = X_base.copy()
    for n in n_grid:
        X[n_col] = float(n)
        X = _refresh_interactions(X)
        yhat = model.predict(X)
        out.append(pd.DataFrame({"n_rate": float(n), "yhat": yhat}))
    return pd.concat(out, axis=0, ignore_index=True)
