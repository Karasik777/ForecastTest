#!/usr/bin/env python3
"""
Plot evaluation results from artifacts/evaluation.csv.
- Produces LaTeX-friendly PNG and PDF figures
- Maps fold_start (time_idx) -> datetime labels via artifacts/time_labels.csv (preferred)
  or via processed parquet as a fallback.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="artifacts/evaluation.csv",
                   help="Path to evaluation.csv produced by evaluate.py")
    p.add_argument("--outdir", default="artifacts",
                   help="Directory to save plots")
    p.add_argument("--data", default="data/processed/merged.parquet",
                   help="Processed parquet (fallback for datetime labels)")
    p.add_argument("--labels", default="artifacts/time_labels.csv",
                   help="Preferred: CSV mapping time_idx -> open_time")
    p.add_argument("--timefmt", default="%Y-%m-%d %H:%M",
                   help="Datetime format for x-axis labels")
    return p.parse_args()


def load_label_map(labels_csv: Path, data_parquet: Path):
    """Return a Series mapping time_idx -> formatted datetime string (or None if unavailable)."""
    # 1) Preferred: pre-computed label map from evaluate.py
    if labels_csv.exists():
        lm = pd.read_csv(labels_csv)
        if "time_idx" in lm.columns and "open_time" in lm.columns:
            s = pd.to_datetime(lm["open_time"], utc=True, errors="coerce")
            return pd.Series(s.values, index=lm["time_idx"].values)
        print(f"⚠️  {labels_csv} found but missing required columns.")

    # 2) Fallback: infer from processed parquet if it contains open_time + time_idx
    try:
        dfp = pd.read_parquet(data_parquet)
        if "time_idx" in dfp.columns and "open_time" in dfp.columns:
            dfp["open_time"] = pd.to_datetime(dfp["open_time"], utc=True, errors="coerce")
            tmp = (dfp.sort_values("open_time")
                        .drop_duplicates("time_idx")[["time_idx", "open_time"]])
            return pd.Series(tmp["open_time"].values, index=tmp["time_idx"].values)
    except Exception as e:
        print(f"⚠️  Could not read {data_parquet} for labels: {e}")

    return None


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # make numeric (handle None/NaN)
    for col in ["mape_naive", "rmse_naive", "mape_tft", "rmse_tft"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build X labels
    fold_starts = df["fold_start"].tolist()
    label_series = load_label_map(Path(args.labels), Path(args.data))
    if label_series is not None:
        def _format_label(fs: int) -> str:
            ts = label_series.get(fs)
            if pd.isna(ts):
                return str(fs)
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.strftime(args.timefmt)

        fold_labels = [_format_label(fs) for fs in fold_starts]
    else:
        print("⚠️  No datetime mapping available; using time_idx integers.")
        fold_labels = [str(fs) for fs in fold_starts]

    # ------------------ 1) MAPE & RMSE per fold ------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(df))
    width = 0.35

    # Left: MAPE
    axes[0].bar(x - width/2, df["mape_naive"], width, label="Naive")
    axes[0].bar(x + width/2, df["mape_tft"], width, label="TFT")
    axes[0].set_title("MAPE by Fold (lower is better)")
    axes[0].set_xlabel("Fold start")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fold_labels, rotation=30, ha="right")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Right: RMSE
    axes[1].bar(x - width/2, df["rmse_naive"], width, label="Naive")
    axes[1].bar(x + width/2, df["rmse_tft"], width, label="TFT")
    axes[1].set_title("RMSE by Fold (lower is better)")
    axes[1].set_xlabel("Fold start")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fold_labels, rotation=30, ha="right")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "eval_mape_rmse.png", dpi=180)
    fig.savefig(outdir / "eval_mape_rmse.pdf")
    plt.close(fig)

    # ------------------ 2) Improvement % (TFT vs Naive) ------------------
    imp_rows = []
    for _, r in df.iterrows():
        mape_imp = np.nan
        rmse_imp = np.nan
        # improvement% = 100 * (naive - tft) / naive
        if not (pd.isna(r.get("mape_naive")) or pd.isna(r.get("mape_tft")) or r.get("mape_naive", 0) == 0):
            mape_imp = 100.0 * (r["mape_naive"] - r["mape_tft"]) / r["mape_naive"]
        if not (pd.isna(r.get("rmse_naive")) or pd.isna(r.get("rmse_tft")) or r.get("rmse_naive", 0) == 0):
            rmse_imp = 100.0 * (r["rmse_naive"] - r["rmse_tft"]) / r["rmse_naive"]
        imp_rows.append((r["fold_start"], mape_imp, rmse_imp))

    imp_df = pd.DataFrame(imp_rows, columns=["fold_start", "mape_improvement_%", "rmse_improvement_%"])

    fig2, ax2 = plt.subplots(figsize=(7.6, 4.2))
    ix = np.arange(len(imp_df))
    ax2.bar(ix - 0.175, imp_df["mape_improvement_%"], width=0.35, label="MAPE improvement %")
    ax2.bar(ix + 0.175, imp_df["rmse_improvement_%"], width=0.35, label="RMSE improvement %")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("TFT Improvement over Naive (higher is better)")
    ax2.set_xlabel("Fold start")
    ax2.set_ylabel("Improvement (%)")
    # use the same datetime labels
    ax2.set_xticks(ix)
    ax2.set_xticklabels(fold_labels, rotation=30, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(outdir / "eval_improvement.png", dpi=180)
    fig2.savefig(outdir / "eval_improvement.pdf")
    plt.close(fig2)

    print(f"Saved plots to {outdir}/eval_mape_rmse.(png|pdf) and {outdir}/eval_improvement.(png|pdf)")


if __name__ == "__main__":
    main()
