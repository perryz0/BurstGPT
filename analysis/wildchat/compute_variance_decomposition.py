#!/usr/bin/env python3
"""
Compute intra-day vs inter-day variance decomposition from existing
wildchat_hourly_windows.csv. No dataset load. Output: wildchat_windowed_variance_stats.json
"""

import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "output"
csv_path = OUT_DIR / "wildchat_hourly_windows.csv"
out_path = OUT_DIR / "wildchat_windowed_variance_stats.json"

def main():
    df = pd.read_csv(csv_path)
    df["hour_bin_dt"] = pd.to_datetime(df["hour_bin_dt"], utc=True)
    df["date"] = df["hour_bin_dt"].dt.date

    # Intra-day: for each day, CV of avg_turn and avg_context across its 24 hourly bins
    daily_cv_turn = []
    daily_cv_context = []
    for date, grp in df.groupby("date"):
        at = grp["avg_turn"]
        ac = grp["avg_context_length"]
        if at.mean() > 0 and len(at) >= 6:
            daily_cv_turn.append(at.std() / at.mean())
        if ac.mean() > 0 and len(ac) >= 6:
            daily_cv_context.append(ac.std() / ac.mean())
    mean_intraday_cv_turn = float(pd.Series(daily_cv_turn).mean())
    std_intraday_cv_turn = float(pd.Series(daily_cv_turn).std())
    mean_intraday_cv_context = float(pd.Series(daily_cv_context).mean())
    std_intraday_cv_context = float(pd.Series(daily_cv_context).std())

    # Inter-day: for each hour_of_day, CV of avg_turn and avg_context across days
    interday_cv_turn = []
    interday_cv_context = []
    for hod, grp in df.groupby("hour_of_day"):
        at = grp["avg_turn"]
        ac = grp["avg_context_length"]
        if at.mean() > 0:
            interday_cv_turn.append(at.std() / at.mean())
        if ac.mean() > 0:
            interday_cv_context.append(ac.std() / ac.mean())
    mean_interday_cv_turn = float(pd.Series(interday_cv_turn).mean())
    mean_interday_cv_context = float(pd.Series(interday_cv_context).mean())

    # Global CV (from same CSV: std/mean across all hourly windows)
    global_cv_turn = float(df["avg_turn"].std() / df["avg_turn"].mean())
    global_cv_context = float(df["avg_context_length"].std() / df["avg_context_length"].mean())

    stats = {
        "global_cv_turn": global_cv_turn,
        "global_cv_context": global_cv_context,
        "mean_intraday_cv_turn": mean_intraday_cv_turn,
        "std_intraday_cv_turn": std_intraday_cv_turn,
        "mean_intraday_cv_context": mean_intraday_cv_context,
        "std_intraday_cv_context": std_intraday_cv_context,
        "mean_interday_cv_turn": mean_interday_cv_turn,
        "mean_interday_cv_context": mean_interday_cv_context,
        "n_days_with_cv": len(daily_cv_turn),
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved", out_path)
    return stats


if __name__ == "__main__":
    main()
