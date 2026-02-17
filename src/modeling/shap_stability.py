from __future__ import annotations
import numpy as np
import pandas as pd

def shap_rank_table_xgb(model, X: pd.DataFrame, focus: list[str]) -> pd.DataFrame:
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # mean absolute SHAP by feature
    imp = np.abs(sv).mean(axis=0)
    tab = pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp})
    tab["rank"] = tab["mean_abs_shap"].rank(ascending=False, method="min").astype(int)

    tab["is_focus"] = tab["feature"].isin(focus)
    return tab.sort_values("rank")
