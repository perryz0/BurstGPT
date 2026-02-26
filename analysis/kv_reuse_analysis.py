#!/usr/bin/env python3
"""
BurstGPT KV-cache reuse potential analysis.
Analyzes whether conversation depth (session turns) varies over time.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "BurstGPT_1.csv"
OUT_DIR = REPO_ROOT / "analysis" / "output"
SESSION_GAP_SEC = 1800  # 30 min: gap above this starts a new session (for inferred sessions)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Step 2: Load & normalize ---
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Columns: {list(df.columns)}")

    # Normalize timestamps: dataset Timestamp is seconds from 0:00:00 on first day
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype(np.float64)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Session ID: in-repo CSV does not have "Session ID" (added in v1.2). Infer sessions.
    has_session_col = "Session ID" in df.columns
    if not has_session_col:
        print("Note: 'Session ID' not in CSV; inferring sessions by temporal proximity (Conversation log only).")
        df["Session ID"] = infer_sessions(df)
    else:
        df["Session ID"] = df["Session ID"].fillna(-1).astype(int)

    # Sanity checks
    n_records = len(df)
    n_sessions = df["Session ID"].nunique()
    t_min, t_max = df["Timestamp"].min(), df["Timestamp"].max()
    print(f"\n--- Sanity checks ---")
    print(f"Total records: {n_records}")
    print(f"Total sessions (or inferred): {n_sessions}")
    print(f"Timestamp range: {t_min:.0f} .. {t_max:.0f} sec ({t_max/86400:.1f} days)")

    # --- Step 3: Session-level statistics ---
    print("\n--- Session-level stats ---")
    session_stats = (
        df.groupby("Session ID", as_index=False)
        .agg(
            start_time=("Timestamp", "min"),
            end_time=("Timestamp", "max"),
            n_turns=("Timestamp", "count"),
        )
        .assign(
            duration_sec=lambda x: x["end_time"] - x["start_time"],
        )
    )
    session_stats["duration_sec"] = session_stats["duration_sec"].clip(lower=0)

    print(f"Mean #turns per session: {session_stats['n_turns'].mean():.2f}")
    print(f"Median #turns: {session_stats['n_turns'].median():.0f}")
    print(f"P90 #turns: {session_stats['n_turns'].quantile(0.90):.0f}")
    print(f"P95 #turns: {session_stats['n_turns'].quantile(0.95):.0f}")
    print(f"Sessions with 1 turn: {(session_stats['n_turns'] == 1).sum()} ({(session_stats['n_turns'] == 1).mean()*100:.1f}%)")
    print(f"Sessions with 2+ turns: {(session_stats['n_turns'] >= 2).sum()}")

    # --- Step 4: Temporal analysis ---
    # Hour of day: 0..23 (integer)
    session_stats["hour"] = (session_stats["start_time"] % 86400) / 3600
    session_stats["hour_int"] = ((session_stats["start_time"] % 86400) // 3600).astype(int)
    session_stats["day_index"] = (session_stats["start_time"] // 86400).astype(int)
    # Time window: 1-hour bins over the full trace (bin by start_time)
    bin_sec = 3600
    session_stats["time_bin"] = (session_stats["start_time"] // bin_sec).astype(int) * bin_sec

    by_hour = (
        session_stats.groupby("hour_int")
        .agg(
            avg_turns=("n_turns", "mean"),
            p90_turns=("n_turns", lambda x: x.quantile(0.90)),
            p95_turns=("n_turns", lambda x: x.quantile(0.95)),
            session_count=("Session ID", "count"),
        )
        .reset_index()
        .rename(columns={"hour_int": "hour"})
    )

    by_bin = (
        session_stats.groupby("time_bin")
        .agg(
            avg_turns=("n_turns", "mean"),
            p90_turns=("n_turns", lambda x: x.quantile(0.90)),
            p95_turns=("n_turns", lambda x: x.quantile(0.95)),
            session_count=("Session ID", "count"),
        )
        .reset_index()
    )

    # --- Step 5: Visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        plt = None

    if plt is not None:
        # 1) Average #turns vs time (line plot) — use time_bin
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(by_bin["time_bin"] / 86400, by_bin["avg_turns"], linewidth=0.8, color="C0")
        ax.set_xlabel("Time (days from start)")
        ax.set_ylabel("Avg #turns per session")
        ax.set_title("Average session depth over time (1-hour bins)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "avg_turns_vs_time.png", dpi=150)
        plt.close()

        # 2) Session count vs time
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(by_bin["time_bin"] / 86400, by_bin["session_count"], alpha=0.6, color="C1")
        ax.set_xlabel("Time (days from start)")
        ax.set_ylabel("Session count")
        ax.set_title("Session count per 1-hour window")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "session_count_vs_time.png", dpi=150)
        plt.close()

        # 3) Distribution of #turns (histogram)
        fig, ax = plt.subplots(figsize=(8, 4))
        max_turns = min(int(session_stats["n_turns"].quantile(0.99)), 50)
        ax.hist(session_stats["n_turns"].clip(upper=max_turns), bins=range(1, max_turns + 2), align="left", edgecolor="black", alpha=0.7)
        ax.set_xlabel("#turns per session")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of session depth (#turns)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "turns_histogram.png", dpi=150)
        plt.close()

        # 4) #turns vs hour of day (box plot)
        fig, ax = plt.subplots(figsize=(12, 4))
        session_stats.boxplot(column="n_turns", by="hour_int", ax=ax)
        ax.set_xlabel("Hour of day (from trace start)")
        ax.set_ylabel("#turns per session")
        ax.set_title("Session depth by hour of day")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "turns_vs_hour_of_day.png", dpi=150)
        plt.close()

        # 5) Bar: avg turns by hour of day (clearer for report)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(by_hour["hour"], by_hour["avg_turns"], width=0.7, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Avg #turns per session")
        ax.set_title("Average session depth by hour of day")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "avg_turns_by_hour.png", dpi=150)
        plt.close()

        print(f"\nPlots saved to: {OUT_DIR}")

    # --- Step 6: Evidence summary (printed) ---
    print("\n--- Evidence check ---")
    turn_std = session_stats["n_turns"].std()
    hour_std = by_hour["avg_turns"].std()
    print(f"Session depth varies across sessions (std #turns): {turn_std:.2f}")
    print(f"Average #turns varies by hour of day (std across hours): {hour_std:.2f}")
    if hour_std > 0.05:
        print("Conclusion: Session depth varies by time of day (KV-reuse potential is time-varying).")
    else:
        print("Conclusion: Modest variation in session depth by hour; see report and plots for details.")

    # Save session stats for report
    session_stats.to_csv(OUT_DIR / "session_stats.csv", index=False)
    by_hour.to_csv(OUT_DIR / "by_hour.csv", index=False)
    by_bin.to_csv(OUT_DIR / "by_time_bin.csv", index=False)

    return OUT_DIR, session_stats, by_hour, by_bin


def infer_sessions(df: pd.DataFrame) -> np.ndarray:
    """Infer session IDs when not present: group Conversation log by temporal proximity."""
    out = np.zeros(len(df), dtype=np.int64)
    session_id = 0
    prev_t = -np.inf
    gap_sec = SESSION_GAP_SEC
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


if __name__ == "__main__":
    main()
