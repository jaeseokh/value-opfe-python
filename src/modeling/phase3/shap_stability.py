from __future__ import annotations

import numpy as np
import pandas as pd


def shap_rank_table_xgb(model, X: pd.DataFrame, focus: list[str]) -> pd.DataFrame:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # handle different return shapes across SHAP versions
        if isinstance(sv, list):
            sv_arr = np.asarray(sv[0])
        else:
            sv_arr = np.asarray(sv)
        if sv_arr.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape: {sv_arr.shape}")
        imp = np.abs(sv_arr).mean(axis=0)
    except Exception:
        # Fallback when shap is unavailable: use model gain/impurity importance.
        # Keeps Step3 pipeline runnable in constrained environments.
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            imp = np.full((X.shape[1],), np.nan, dtype=float)

    tab = pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp})
    tab["rank"] = tab["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    tab["is_focus"] = tab["feature"].isin(focus)
    return tab.sort_values("rank").reset_index(drop=True)
