from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        # default to CSV for ".csv" and unknown suffixes
        df.to_csv(path, index=False)


def fig_delta_eonr(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = df.get("lolo_group", pd.Series(range(len(df)))).astype(str)
    y = pd.to_numeric(df.get("abs_delta_eonr"), errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y)
    ax.set_xlabel("LOLO holdout group")
    ax.set_ylabel("abs(delta EONR)")
    ax.set_title("Step2: Absolute EONR Gap by Holdout")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_focus_rank_boxplot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    if df.empty or "feature" not in df.columns or "rank" not in df.columns:
        ax.text(0.5, 0.5, "No focus-rank data", ha="center", va="center")
        ax.set_axis_off()
    else:
        features = sorted(df["feature"].astype(str).unique().tolist())
        data = [pd.to_numeric(df.loc[df["feature"] == f, "rank"], errors="coerce").dropna() for f in features]
        ax.boxplot(data, tick_labels=features)
        ax.set_ylabel("SHAP rank (lower is more important)")
        ax.set_title("Step3: Focus-Feature Rank Stability")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

