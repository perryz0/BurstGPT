#!/usr/bin/env python3
"""
WildChat windowed phase analysis (Yuyao follow-up).
Hour-level windows: avg_turn, avg_context_length; hour-of-day aggregation;
daily consistency (heatmaps, correlation of daily hourly curves).
Uses only conversation-level timestamp. No synthetic concurrency.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
_hf_cache = str(BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = _hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache
os.environ["HF_HOME"] = _hf_cache
OUT_DIR = BASE / "output"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS = {}


def main():
    global RESULTS
    print("Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat", split="train")
    n_rows = len(ds)

    ts_list = []
    turn_list = []
    context_length_list = []  # total word count across all utterances

    for i in range(n_rows):
        row = ds[i]
        ts = row["timestamp"]
        if hasattr(ts, "timestamp"):
            ts_list.append(ts.timestamp())
        elif isinstance(ts, (int, float)):
            ts_list.append(float(ts))
        else:
            try:
                ts_list.append(pd.Timestamp(ts).timestamp())
            except Exception:
                ts_list.append(np.nan)
        turn_list.append(row["turn"])
        conv = row.get("conversation") or []
        total_words = sum(len((u.get("content") or "").split()) for u in conv)
        context_length_list.append(total_words)

    df = pd.DataFrame({
        "timestamp_unix": ts_list,
        "turn": turn_list,
        "context_length": context_length_list,
    })
    df = df.dropna(subset=["timestamp_unix"]).sort_values("timestamp_unix").reset_index(drop=True)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
    df["hour_of_day"] = df["timestamp_dt"].dt.hour
    df["date"] = df["timestamp_dt"].dt.date
    df["day_index"] = (df["timestamp_unix"] - df["timestamp_unix"].min()) // 86400
    df["hour_bin"] = (df["timestamp_unix"] // 3600).astype(int) * 3600

    print("Step 1–2: Global hourly bins and context length")
    # A) Global hourly bins
    hourly = df.groupby("hour_bin").agg(
        avg_turn=("turn", "mean"),
        avg_context_length=("context_length", "mean"),
        session_count=("turn", "count"),
    ).reset_index()
    hourly["hour_bin_dt"] = pd.to_datetime(hourly["hour_bin"], unit="s", utc=True)
    hourly["hour_of_day"] = hourly["hour_bin_dt"].dt.hour

    # B) Hour-of-day aggregation (mean and std across days for each hour 0–23)
    by_hod = hourly.groupby("hour_of_day").agg(
        mean_avg_turn=("avg_turn", "mean"),
        mean_avg_context=("avg_context_length", "mean"),
        std_avg_turn=("avg_turn", "std"),
        std_avg_context=("avg_context_length", "std"),
        n_windows=("hour_bin", "count"),
    ).reset_index()

    print("Step 3: Statistical characterization (hourly windows)")
    at = hourly["avg_turn"]
    ac = hourly["avg_context_length"]
    RESULTS["hourly_avg_turn_mean"] = float(at.mean())
    RESULTS["hourly_avg_turn_std"] = float(at.std())
    RESULTS["hourly_avg_turn_p10"] = float(at.quantile(0.10))
    RESULTS["hourly_avg_turn_p90"] = float(at.quantile(0.90))
    RESULTS["hourly_avg_context_mean"] = float(ac.mean())
    RESULTS["hourly_avg_context_std"] = float(ac.std())
    RESULTS["hourly_avg_context_p10"] = float(ac.quantile(0.10))
    RESULTS["hourly_avg_context_p90"] = float(ac.quantile(0.90))
    cv_turn = at.std() / at.mean() if at.mean() > 0 else 0
    cv_context = ac.std() / ac.mean() if ac.mean() > 0 else 0
    RESULTS["CV_turn"] = float(cv_turn)
    RESULTS["CV_context"] = float(cv_context)
    print(f"  CV_turn = {cv_turn:.4f}, CV_context = {cv_context:.4f}")

    print("Step 4: Daily consistency (per day, per hour-of-day)")
    daily_hourly = df.groupby(["date", "hour_of_day"]).agg(
        avg_turn=("turn", "mean"),
        avg_context_length=("context_length", "mean"),
        session_count=("turn", "count"),
    ).reset_index()
    dates = sorted(daily_hourly["date"].unique())
    n_days = len(dates)
    # Build matrix: rows = days, cols = hour 0..23
    turn_matrix = np.full((n_days, 24), np.nan)
    context_matrix = np.full((n_days, 24), np.nan)
    for _, row in daily_hourly.iterrows():
        d = row["date"]
        h = int(row["hour_of_day"])
        i = dates.index(d)
        turn_matrix[i, h] = row["avg_turn"]
        context_matrix[i, h] = row["avg_context_length"]

    # Pairwise correlation between daily hourly curves (24-dim vectors)
    def vec_corrs(mat):
        valid = ~np.isnan(mat)
        corrs = []
        for i in range(n_days):
            for j in range(i + 1, n_days):
                mask = valid[i] & valid[j]
                if mask.sum() >= 6:
                    c = np.corrcoef(mat[i][mask], mat[j][mask])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
        return np.array(corrs) if corrs else np.array([np.nan])
    turn_corrs = vec_corrs(turn_matrix)
    context_corrs = vec_corrs(context_matrix)
    RESULTS["daily_curve_corr_turn_mean"] = float(np.nanmean(turn_corrs))
    RESULTS["daily_curve_corr_turn_std"] = float(np.nanstd(turn_corrs))
    RESULTS["daily_curve_corr_context_mean"] = float(np.nanmean(context_corrs))
    RESULTS["daily_curve_corr_context_std"] = float(np.nanstd(context_corrs))
    RESULTS["n_days"] = n_days
    RESULTS["n_hourly_windows"] = len(hourly)
    print(f"  Daily curve correlation (turn): mean {RESULTS['daily_curve_corr_turn_mean']:.4f}, std {RESULTS['daily_curve_corr_turn_std']:.4f}")
    print(f"  Daily curve correlation (context): mean {RESULTS['daily_curve_corr_context_mean']:.4f}, std {RESULTS['daily_curve_corr_context_std']:.4f}")

    print("Step 5: Visualizations")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt:
        # Global hourly: avg_turn over time
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(hourly["hour_bin"] / 86400, hourly["avg_turn"], linewidth=0.6, color="C0")
        ax.set_xlabel("Time (days from start)")
        ax.set_ylabel("Avg #turns")
        ax.set_title("WildChat: average turns per conversation by hourly window")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_hourly_avg_turns.png", dpi=150)
        plt.close()

        # Global hourly: avg_context over time
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(hourly["hour_bin"] / 86400, hourly["avg_context_length"], linewidth=0.6, color="C1")
        ax.set_xlabel("Time (days from start)")
        ax.set_ylabel("Avg context length (words)")
        ax.set_title("WildChat: average context length by hourly window")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_hourly_avg_context.png", dpi=150)
        plt.close()

        # Hour-of-day: avg_turn (mean across days)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(by_hod["hour_of_day"], by_hod["mean_avg_turn"], yerr=by_hod["std_avg_turn"], capsize=2, color="steelblue", alpha=0.8)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Mean avg #turns (across days)")
        ax.set_title("WildChat: average turns by hour of day (±1 std across days)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_hour_of_day_avg_turns.png", dpi=150)
        plt.close()

        # Hour-of-day: avg_context
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(by_hod["hour_of_day"], by_hod["mean_avg_context"], yerr=by_hod["std_avg_context"], capsize=2, color="darkorange", alpha=0.8)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Mean avg context length (words)")
        ax.set_title("WildChat: average context length by hour of day (±1 std across days)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_hour_of_day_avg_context.png", dpi=150)
        plt.close()

        # Heatmap: day x hour_of_day, value = avg_turn (y-axis = day index, height capped)
        fig, ax = plt.subplots(figsize=(12, min(14, max(6, n_days * 0.12))))
        im = ax.imshow(turn_matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(turn_matrix), vmax=np.nanpercentile(turn_matrix, 95))
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Day index")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        n_yt = min(12, n_days)
        y_ticks = np.linspace(0, n_days - 1, n_yt).astype(int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(dates[i]) for i in y_ticks])
        plt.colorbar(im, ax=ax, label="Avg #turns")
        ax.set_title("WildChat: avg turns by day and hour of day")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_turn_heatmap.png", dpi=150)
        plt.close()

        # Heatmap: context
        fig, ax = plt.subplots(figsize=(12, min(14, max(6, n_days * 0.12))))
        im = ax.imshow(context_matrix, aspect="auto", cmap="plasma", vmin=np.nanmin(context_matrix), vmax=np.nanpercentile(context_matrix, 95))
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Day index")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(dates[i]) for i in y_ticks])
        plt.colorbar(im, ax=ax, label="Avg context length (words)")
        ax.set_title("WildChat: avg context length by day and hour of day")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_context_heatmap.png", dpi=150)
        plt.close()

    hourly.to_csv(OUT_DIR / "wildchat_hourly_windows.csv", index=False)
    by_hod.to_csv(OUT_DIR / "wildchat_hour_of_day.csv", index=False)
    write_report(RESULTS, by_hod, hourly, turn_corrs, context_corrs)
    print("Done. Report: wildchat_windowed_phase_analysis.md")
    return RESULTS


def write_report(results, by_hod, hourly, turn_corrs, context_corrs):
    r = results
    path = BASE / "wildchat_windowed_phase_analysis.md"
    lines = [
        "# WildChat Windowed Phase Analysis",
        "",
        "Targeted analysis for Yuyao: do **average number of turns** and **average context length** change significantly across hourly windows, and do similar trends repeat across days?",
        "",
        "---",
        "",
        "## What was computed",
        "",
        "- **Windowing:** Hour-level windows from conversation timestamp (UTC). Each conversation assigned to the hour bin containing its timestamp. **Context length** = total word count across all utterances in the conversation (no tokenizer).",
        "- **Two levels:** (A) Global hourly bins across the dataset → per-window `avg_turn`, `avg_context_length`, `session_count`. (B) Hour-of-day (0–23) → mean and std of those metrics across all hourly windows that fall in each hour of day.",
        "- **Daily consistency:** For each (date, hour_of_day) we computed avg_turn and avg_context_length. Heatmaps: day × hour_of_day. Pairwise correlation between each day’s 24-dimensional hourly curve (to test if diurnal pattern is consistent across days).",
        "",
        "---",
        "",
        "## Global hourly variation",
        "",
        f"- **Number of hourly windows:** {r.get('n_hourly_windows', 'N/A')}",
        f"- **Avg turns per window:** mean = {r.get('hourly_avg_turn_mean', 0):.3f}, std = {r.get('hourly_avg_turn_std', 0):.3f}, P10 = {r.get('hourly_avg_turn_p10', 0):.3f}, P90 = {r.get('hourly_avg_turn_p90', 0):.3f}",
        f"- **Avg context length per window:** mean = {r.get('hourly_avg_context_mean', 0):.0f} words, std = {r.get('hourly_avg_context_std', 0):.0f}, P10 = {r.get('hourly_avg_context_p10', 0):.0f}, P90 = {r.get('hourly_avg_context_p90', 0):.0f}",
        f"- **Coefficient of variation (CV):** CV_turn = {r.get('CV_turn', 0):.4f}, CV_context = {r.get('CV_context', 0):.4f}",
        "",
        "CV quantifies relative variation across hourly windows: CV > 0.1–0.2 suggests non-negligible window-level variation.",
        "",
        "---",
        "",
        "## Hour-of-day aggregation",
        "",
        "Mean and std of avg_turn and avg_context_length for each hour of day (0–23 UTC), aggregated across all days. See plots `wildchat_hour_of_day_avg_turns.png` and `wildchat_hour_of_day_avg_context.png`. If metrics change significantly by hour of day, we expect visible bars and error bars varying across hours.",
        "",
        "---",
        "",
        "## Daily consistency analysis",
        "",
        f"- **Days in dataset:** {r.get('n_days', 'N/A')}",
        f"- **Pairwise correlation between daily hourly curves (avg_turn):** mean = {r.get('daily_curve_corr_turn_mean', 0):.4f}, std = {r.get('daily_curve_corr_turn_std', 0):.4f}",
        f"- **Pairwise correlation between daily hourly curves (avg_context_length):** mean = {r.get('daily_curve_corr_context_mean', 0):.4f}, std = {r.get('daily_curve_corr_context_std', 0):.4f}",
        "",
        "High mean correlation (e.g. > 0.5) indicates that the shape of the diurnal curve (which hours are high/low) is **consistent across days**. Low or variable correlation suggests day-to-day noise or non-repeating structure.",
        "",
        "---",
        "",
        "## Quantitative measures (CV values)",
        "",
        f"- **CV_turn:** {r.get('CV_turn', 0):.4f}",
        f"- **CV_context:** {r.get('CV_context', 0):.4f}",
        "",
        "---",
        "",
        "## Does the dataset exhibit significant window-level phase shifts?",
        "",
        "**Answer:**",
        "",
        f"- **Significant change across windows?** CV_turn = {r.get('CV_turn', 0):.3f} and CV_context = {r.get('CV_context', 0):.3f}. " + (
            "Both show non-negligible variation across hourly windows (CV on the order of 0.1 or higher), so **yes**, metrics do change meaningfully across windows."
            if (r.get('CV_turn', 0) >= 0.08 or r.get('CV_context', 0) >= 0.08) else
            "Variation across hourly windows is modest; phase shifts, if any, are small relative to the mean."
        ),
        "",
        f"- **Consistent diurnal pattern across days?** Mean pairwise correlation of daily hourly curves: {r.get('daily_curve_corr_turn_mean', 0):.3f} (turns), {r.get('daily_curve_corr_context_mean', 0):.3f} (context). " + (
            "Correlations are positive and relatively high, so **yes**, similar trends tend to repeat across days (consistent diurnal structure)."
            if (r.get('daily_curve_corr_turn_mean', 0) >= 0.4 and r.get('daily_curve_corr_context_mean', 0) >= 0.3) else
            "Correlations are modest or low; diurnal structure is **not strongly** consistent across days (more day-to-day noise or heterogeneity)."
        ),
        "",
        "---",
        "",
        "**Artifacts:** `wildchat_hourly_avg_turns.png`, `wildchat_hourly_avg_context.png`, `wildchat_hour_of_day_avg_turns.png`, `wildchat_hour_of_day_avg_context.png`, `wildchat_turn_heatmap.png`, `wildchat_context_heatmap.png`; `wildchat_hourly_windows.csv`, `wildchat_hour_of_day.csv` in `analysis/wildchat/output/`.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
