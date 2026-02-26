#!/usr/bin/env python3
"""
BurstGPT KV-reuse analysis: robustness extensions.
- Sensitivity to session-gap heuristic (15m, 30m, 60m).
- Fraction-based metrics (frac ≥2 turns, frac ≥3 turns) by hour and over time.
- Sparse-bin filtering (min 100 sessions) for trend robustness.
All outputs use new filenames; does not overwrite baseline results.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "BurstGPT_1.csv"
OUT_DIR = REPO_ROOT / "analysis" / "output"
MIN_SESSION_COUNT = 100  # for sparse-bin filtering


def infer_sessions(df: pd.DataFrame, gap_sec: int) -> np.ndarray:
    """Infer session IDs: group Conversation log by temporal proximity (gap_sec)."""
    out = np.zeros(len(df), dtype=np.int64)
    session_id = 0
    prev_t = -np.inf
    log_type_col = "Log Type" if "Log Type" in df.columns else None
    for i in range(len(df)):
        row = df.iloc[i]
        t = row["Timestamp"]
        log_type = row[log_type_col] if log_type_col else "Conversation log"
        if log_type != "Conversation log":
            session_id += 1
            out[i] = session_id
            prev_t = t
            continue
        if t - prev_t > gap_sec:
            session_id += 1
        out[i] = session_id
        prev_t = t
    return out


def run_gap(df: pd.DataFrame, gap_sec: int, label: str) -> tuple:
    """Compute session stats, by_hour, by_bin for a given session gap. Returns (session_stats, by_hour, by_bin)."""
    df = df.copy()
    df["Session ID"] = infer_sessions(df, gap_sec)
    session_stats = (
        df.groupby("Session ID", as_index=False)
        .agg(
            start_time=("Timestamp", "min"),
            end_time=("Timestamp", "max"),
            n_turns=("Timestamp", "count"),
        )
        .assign(duration_sec=lambda x: (x["end_time"] - x["start_time"]).clip(lower=0))
    )
    session_stats["hour_int"] = ((session_stats["start_time"] % 86400) // 3600).astype(int)
    session_stats["time_bin"] = (session_stats["start_time"] // 3600).astype(int) * 3600
    session_stats["ge2"] = (session_stats["n_turns"] >= 2).astype(int)
    session_stats["ge3"] = (session_stats["n_turns"] >= 3).astype(int)

    by_hour = (
        session_stats.groupby("hour_int")
        .agg(
            avg_turns=("n_turns", "mean"),
            session_count=("Session ID", "count"),
            frac_ge2=("ge2", "mean"),
            frac_ge3=("ge3", "mean"),
        )
        .reset_index()
        .rename(columns={"hour_int": "hour"})
    )
    by_bin = (
        session_stats.groupby("time_bin")
        .agg(
            avg_turns=("n_turns", "mean"),
            session_count=("Session ID", "count"),
            frac_ge2=("ge2", "mean"),
            frac_ge3=("ge3", "mean"),
        )
        .reset_index()
    )
    return session_stats, by_hour, by_bin


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    gaps = [
        (900, "15m"),
        (1800, "30m"),
        (3600, "60m"),
    ]
    results = {}
    for gap_sec, label in gaps:
        print(f"Computing gap = {label}...")
        session_stats, by_hour, by_bin = run_gap(df, gap_sec, label)
        results[label] = {
            "session_stats": session_stats,
            "by_hour": by_hour,
            "by_bin": by_bin,
        }
        by_hour.to_csv(OUT_DIR / f"by_hour_{label}.csv", index=False)
        by_bin.to_csv(OUT_DIR / f"by_bin_{label}.csv", index=False)
        n_sessions = len(session_stats)
        frac_ge2 = (session_stats["n_turns"] >= 2).mean()
        frac_ge3 = (session_stats["n_turns"] >= 3).mean()
        print(f"  sessions={n_sessions}, frac≥2={frac_ge2:.4f}, frac≥3={frac_ge3:.4f}, avg_turns={session_stats['n_turns'].mean():.3f}")

    # Sensitivity summary table
    sensitivity_rows = []
    for label in ["15m", "30m", "60m"]:
        ss = results[label]["session_stats"]
        by_h = results[label]["by_hour"]
        sensitivity_rows.append({
            "gap": label,
            "n_sessions": len(ss),
            "frac_ge2_turns": (ss["n_turns"] >= 2).mean(),
            "frac_ge3_turns": (ss["n_turns"] >= 3).mean(),
            "avg_turns": ss["n_turns"].mean(),
            "avg_turns_std_over_hours": by_h["avg_turns"].std(),
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(OUT_DIR / "sensitivity_session_gap.csv", index=False)
    print("\nSensitivity summary (sensitivity_session_gap.csv):")
    print(sensitivity_df.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    # --- Plot 1: Sensitivity — avg_turns by hour for 15m, 30m, 60m ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, color in [("15m", "C0"), ("30m", "C1"), ("60m", "C2")]:
        by_h = results[label]["by_hour"]
        ax.plot(by_h["hour"], by_h["avg_turns"], label=f"gap={label}", color=color, linewidth=1.2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg #turns per session")
    ax.set_title("Sensitivity: average session depth by hour (session-gap heuristic)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_avg_turns_by_hour.png", dpi=150)
    plt.close()

    # --- Plot 2: Fraction ≥2 turns by hour (30m baseline) ---
    by_h = results["30m"]["by_hour"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_h["hour"], by_h["frac_ge2"] * 100, width=0.7, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraction of sessions with ≥2 turns (%)")
    ax.set_title("Fraction of multi-turn sessions by hour of day (30m gap)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraction_ge2_by_hour.png", dpi=150)
    plt.close()

    # --- Plot 3: Fraction ≥2 turns vs time (30m, 1-hour bins) ---
    by_bin = results["30m"]["by_bin"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_bin["time_bin"] / 86400, by_bin["frac_ge2"] * 100, linewidth=0.8, color="C0")
    ax.set_xlabel("Time (days from start)")
    ax.set_ylabel("Fraction of sessions with ≥2 turns (%)")
    ax.set_title("Fraction of multi-turn sessions over time (1-hour bins, 30m gap)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraction_ge2_vs_time.png", dpi=150)
    plt.close()

    # --- Plot 4: Fraction ≥3 turns by hour ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_h["hour"], by_h["frac_ge3"] * 100, width=0.7, color="darkorange", edgecolor="brown", alpha=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraction of sessions with ≥3 turns (%)")
    ax.set_title("Fraction of deep sessions (≥3 turns) by hour of day (30m gap)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraction_ge3_by_hour.png", dpi=150)
    plt.close()

    # --- Plot 5: Fraction ≥3 turns vs time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_bin["time_bin"] / 86400, by_bin["frac_ge3"] * 100, linewidth=0.8, color="C1")
    ax.set_xlabel("Time (days from start)")
    ax.set_ylabel("Fraction of sessions with ≥3 turns (%)")
    ax.set_title("Fraction of deep sessions over time (1-hour bins, 30m gap)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraction_ge3_vs_time.png", dpi=150)
    plt.close()

    # --- Sparse-bin filtering: keep bins with session_count >= MIN_SESSION_COUNT ---
    by_bin_f = by_bin[by_bin["session_count"] >= MIN_SESSION_COUNT].copy()
    print(f"\nSparse-bin filter: {len(by_bin)} → {len(by_bin_f)} bins (min {MIN_SESSION_COUNT} sessions)")

    # --- Plot 6: Avg turns vs time (filtered) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_bin_f["time_bin"] / 86400, by_bin_f["avg_turns"], linewidth=0.8, color="C0")
    ax.set_xlabel("Time (days from start)")
    ax.set_ylabel("Avg #turns per session")
    ax.set_title(f"Average session depth over time (1-hour bins, ≥{MIN_SESSION_COUNT} sessions per bin)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "avg_turns_vs_time_min100.png", dpi=150)
    plt.close()

    # --- Plot 7: Fraction ≥2 vs time (filtered) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_bin_f["time_bin"] / 86400, by_bin_f["frac_ge2"] * 100, linewidth=0.8, color="C0")
    ax.set_xlabel("Time (days from start)")
    ax.set_ylabel("Fraction of sessions with ≥2 turns (%)")
    ax.set_title(f"Fraction of multi-turn sessions over time (≥{MIN_SESSION_COUNT} sessions per bin)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraction_ge2_vs_time_min100.png", dpi=150)
    plt.close()

    # Persist filtered by_bin for report
    by_bin_f.to_csv(OUT_DIR / "by_bin_30m_min100.csv", index=False)
    print(f"Plots and CSVs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
