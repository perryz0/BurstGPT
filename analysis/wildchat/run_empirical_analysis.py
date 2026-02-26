#!/usr/bin/env python3
"""
Empirical workload analysis of WildChat (allenai/WildChat).
Produces distributions, timestamp validation, arrival process, model-based concurrency.
All values computed from the dataset; no fabrication.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Use project-local cache for HuggingFace (avoids ~/.cache permission issues)
BASE = Path(__file__).resolve().parent
_hf_cache = str(BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = _hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache
os.environ["HF_HOME"] = _hf_cache
OUT_DIR = BASE / "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Will be populated after load
stats = {}


def main():
    global stats
    print("Step 1 — Load Dataset")
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat", split="train")
    n_rows = len(ds)
    print(f"  Dataset size: {n_rows}")
    print(f"  Column names: {ds.column_names}")

    # Inspect types and first row
    first = ds[0]
    print("  First row keys:", list(first.keys()))
    for k in ["conversation_id", "conversation", "turn", "timestamp", "model"]:
        v = first.get(k)
        print(f"    {k}: type={type(v).__name__}, sample={repr(v)[:80] if v is not None else 'None'}...")

    # Build DataFrame: extract needed columns without holding full conversation text
    ts_list = []
    turn_list = []
    conv_id_list = []
    model_list = []
    turn_from_conv_list = []
    words_per_turn_list = []  # list of lists, one per conversation
    model_per_conv = []

    for i in range(n_rows):
        row = ds[i]
        conv_id_list.append(row["conversation_id"])
        turn_list.append(row["turn"])
        model_list.append(row["model"])
        ts = row["timestamp"]
        if hasattr(ts, "timestamp"):
            ts_list.append(ts.timestamp())
        elif isinstance(ts, (int, float)):
            ts_list.append(float(ts))
        else:
            try:
                dt = pd.Timestamp(ts)
                ts_list.append(dt.timestamp())
            except Exception:
                ts_list.append(np.nan)
        conv = row.get("conversation") or []
        turn_from_conv_list.append(len(conv) // 2 if conv else 0)  # user-assistant pairs
        words = []
        for ut in conv:
            c = ut.get("content") or ""
            words.append(len(c.split()))
        words_per_turn_list.append(words)

    df = pd.DataFrame({
        "conversation_id": conv_id_list,
        "turn": turn_list,
        "timestamp_unix": ts_list,
        "model": model_list,
        "turn_from_conv": turn_from_conv_list,
    })
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
    df = df.dropna(subset=["timestamp_unix"]).sort_values("timestamp_unix").reset_index(drop=True)
    n_valid = len(df)
    stats["n_rows"] = n_rows
    stats["n_valid_ts"] = n_valid
    stats["columns"] = ds.column_names

    # Words per turn: per-conversation average
    avg_words = [np.mean(w) if w else 0 for w in words_per_turn_list]
    df["avg_words_per_turn"] = [avg_words[i] for i in range(n_valid)]

    print(f"  Valid rows (non-null timestamp): {n_valid}")
    print("  Sample timestamp_dt:", df["timestamp_dt"].iloc[0])
    print("  Sample timestamp_unix:", df["timestamp_unix"].iloc[0])

    # --- Step 2 — Validate Timestamps ---
    print("\nStep 2 — Validate Timestamps")
    t_min = df["timestamp_unix"].min()
    t_max = df["timestamp_unix"].max()
    span_days = (t_max - t_min) / 86400
    stats["ts_min"] = t_min
    stats["ts_max"] = t_max
    stats["ts_span_days"] = span_days
    stats["ts_min_dt"] = str(pd.Timestamp.fromtimestamp(t_min, tz="UTC"))
    stats["ts_max_dt"] = str(pd.Timestamp.fromtimestamp(t_max, tz="UTC"))
    print(f"  min timestamp: {t_min} ({stats['ts_min_dt']})")
    print(f"  max timestamp: {t_max} ({stats['ts_max_dt']})")
    print(f"  span: {span_days:.1f} days")

    df["hour"] = df["timestamp_dt"].dt.hour
    df["dayofweek"] = df["timestamp_dt"].dt.dayofweek
    df["date"] = df["timestamp_dt"].dt.date
    dup = df["timestamp_unix"].duplicated(keep=False)
    stats["n_duplicate_ts"] = dup.sum()
    stats["strictly_increasing"] = (df["timestamp_unix"].diff().dropna() >= 0).all()
    print(f"  Duplicate timestamps: {dup.sum()}")
    print(f"  Strictly increasing (sorted): {stats['strictly_increasing']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["hour"], bins=24, range=(0, 24), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Conversations")
        ax.set_title("WildChat: conversations by hour of day (UTC)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_hour_hist.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["dayofweek"], bins=7, range=(-0.5, 6.5), edgecolor="black", alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Conversations")
        ax.set_title("WildChat: conversations by day of week")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_dow_hist.png", dpi=150)
        plt.close()

        conv_per_day = df.groupby("date").size()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(conv_per_day)), conv_per_day.values, linewidth=0.8)
        ax.set_xlabel("Day index")
        ax.set_ylabel("Conversations per day")
        ax.set_title("WildChat: conversations per day")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_conv_per_day.png", dpi=150)
        plt.close()

    # --- Step 3 — Session Length Distribution ---
    print("\nStep 3 — Session Length Distribution")
    turn = df["turn"]
    stats["turn_mean"] = float(turn.mean())
    stats["turn_median"] = float(turn.median())
    stats["turn_p90"] = float(turn.quantile(0.90))
    stats["turn_p95"] = float(turn.quantile(0.95))
    stats["turn_p99"] = float(turn.quantile(0.99))
    stats["frac_ge2"] = float((turn >= 2).mean())
    stats["frac_ge3"] = float((turn >= 3).mean())
    stats["frac_ge5"] = float((turn >= 5).mean())
    stats["frac_ge10"] = float((turn >= 10).mean())
    for k, v in stats.items():
        if k.startswith("turn_") or k.startswith("frac_"):
            print(f"  {k}: {v}")

    turn_from_conv = df["turn_from_conv"]
    stats["turn_field_vs_conv_corr"] = float(turn.corr(turn_from_conv)) if len(turn) > 1 else 0
    stats["turn_field_vs_conv_agree"] = float((turn == turn_from_conv).mean())
    print(f"  turn field vs manual count correlation: {stats['turn_field_vs_conv_corr']:.4f}")
    print(f"  turn field == manual count (frac): {stats['turn_field_vs_conv_agree']:.4f}")

    if plt:
        fig, ax = plt.subplots(figsize=(8, 4))
        max_t = min(int(turn.quantile(0.99)), 50)
        ax.hist(turn.clip(upper=max_t), bins=range(1, max_t + 2), align="left", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Turn count")
        ax.set_ylabel("Conversations")
        ax.set_title("WildChat: session length (turns) distribution")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_session_length_hist.png", dpi=150)
        plt.close()

        # CCDF
        turn_sorted = np.sort(turn)
        ccdf = 1 - np.arange(1, len(turn_sorted) + 1) / len(turn_sorted)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(turn_sorted, ccdf, linewidth=0.8)
        ax.set_xlabel("Turn count")
        ax.set_ylabel("P(Turns >= x)")
        ax.set_title("WildChat: CCDF of session length")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_session_length_ccdf.png", dpi=150)
        plt.close()

    # --- Step 4 — Arrival Process ---
    print("\nStep 4 — Arrival Process")
    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    inter_arrival = df["timestamp_unix"].diff().dropna()
    inter_arrival = inter_arrival[inter_arrival > 0]
    stats["ia_mean"] = float(inter_arrival.mean())
    stats["ia_median"] = float(inter_arrival.median())
    stats["ia_p95"] = float(inter_arrival.quantile(0.95))
    stats["ia_min"] = float(inter_arrival.min())
    stats["ia_max"] = float(inter_arrival.max())
    print(f"  Inter-arrival (s): mean={stats['ia_mean']:.2f}, median={stats['ia_median']:.2f}, p95={stats['ia_p95']:.2f}")

    # Arrivals per minute (rolling) and per hour
    df["bin_1min"] = (df["timestamp_unix"] // 60).astype(int) * 60
    df["bin_1h"] = (df["timestamp_unix"] // 3600).astype(int) * 3600
    per_min = df.groupby("bin_1min").size()
    per_hour = df.groupby("bin_1h").size()
    stats["arrival_rate_per_min_mean"] = float(per_min.mean())
    stats["arrival_rate_per_hour_mean"] = float(per_hour.mean())
    stats["arrival_rate_per_hour_std"] = float(per_hour.std())

    if plt:
        fig, ax = plt.subplots(figsize=(8, 4))
        ia_pos = inter_arrival[inter_arrival < inter_arrival.quantile(0.99)]
        ax.hist(np.log10(ia_pos + 1), bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("log10(inter-arrival time + 1) (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("WildChat: inter-arrival time distribution")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_interarrival_hist.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(per_hour.index / 86400, per_hour.values, linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Time (days from start)")
        ax.set_ylabel("Conversations per hour")
        ax.set_title("WildChat: arrival rate (conversations per hour)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_arrival_rate.png", dpi=150)
        plt.close()

    # By hour of day
    by_hour = df.groupby("hour").size()
    stats["arrival_by_hour_std"] = float(by_hour.std())
    by_dow = df.groupby("dayofweek").size()
    stats["arrival_by_dow_std"] = float(by_dow.std())

    # --- Step 5 — Model-based concurrency (event-based sweep) ---
    print("\nStep 5 — Model-based concurrency")
    for mult, label in [(10, "10s"), (30, "30s")]:
        df["duration"] = df["turn"] * mult
        df["start_time"] = df["timestamp_unix"] - df["duration"]
        events = []
        for _, row in df[["start_time", "timestamp_unix"]].iterrows():
            s, e = row["start_time"], row["timestamp_unix"]
            events.append((s, 1))
            events.append((e + 1e-6, -1))
        events.sort(key=lambda x: x[0])
        t_vals = []
        c_vals = []
        cur = 0
        for t, delta in events:
            cur += delta
            t_vals.append(t)
            c_vals.append(cur)
        t_arr = np.array(t_vals)
        c_arr = np.array(c_vals)
        stats[f"concurrency_{label}_peak"] = float(c_arr.max())
        stats[f"concurrency_{label}_mean"] = float(np.mean(c_arr))
        stats[f"concurrency_{label}_median"] = float(np.median(c_arr))
        print(f"  {label}/turn: peak={c_arr.max():.0f}, mean={np.mean(c_arr):.1f}")

        if plt:
            # Downsample for plot (events can be huge)
            n_plot = min(len(t_arr), 8000)
            idx = np.linspace(0, len(t_arr) - 1, n_plot).astype(int)
            t_min_g = df["start_time"].min()
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot((t_arr[idx] - t_min_g) / 86400, c_arr[idx], linewidth=0.5, alpha=0.8)
            ax.set_xlabel("Time (days from start)")
            ax.set_ylabel("Concurrent sessions")
            ax.set_title(f"WildChat: model-based concurrency (duration = turn × {mult}s)")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"wildchat_concurrency_{label}.png", dpi=150)
            plt.close()

    # --- Step 6 — Structural (words per turn, by model) ---
    print("\nStep 6 — Structural characteristics")
    stats["avg_words_per_turn_overall"] = float(df["avg_words_per_turn"].mean())
    by_model = df.groupby("model").agg(
        turn_mean=("turn", "mean"),
        turn_median=("turn", "median"),
        count=("conversation_id", "count"),
        avg_words=("avg_words_per_turn", "mean"),
    ).reset_index()
    stats["models"] = by_model["model"].tolist()
    stats["turn_by_model"] = by_model.set_index("model")["turn_mean"].to_dict()
    stats["count_by_model"] = by_model.set_index("model")["count"].to_dict()

    if plt:
        fig, ax = plt.subplots(figsize=(8, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]["turn"]
            ax.hist(subset.clip(upper=30), bins=range(1, 32), alpha=0.5, label=str(model)[:20], density=True)
        ax.set_xlabel("Turn count")
        ax.set_ylabel("Density")
        ax.set_title("WildChat: session length distribution by model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "wildchat_turn_distribution_by_model.png", dpi=150)
        plt.close()

    def _serialize(v):
        if isinstance(v, (int, float, str, type(None))):
            return v
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, (list, tuple)):
            return [_serialize(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _serialize(x) for k, x in v.items()}
        return str(v)

    with open(OUT_DIR / "wildchat_empirical_stats.json", "w") as f:
        json.dump({k: _serialize(v) for k, v in stats.items()}, f, indent=2)
    by_model.to_csv(OUT_DIR / "wildchat_by_model.csv", index=False)

    # --- Step 7 — Write empirical report ---
    write_empirical_report(stats, OUT_DIR)
    print(f"\nArtifacts saved to {OUT_DIR}")
    return stats, OUT_DIR


def write_empirical_report(stats, out_dir):
    s = stats
    lines = [
        "# WildChat Empirical Workload Analysis (v1)",
        "",
        "Evidence-based summary from running `run_empirical_analysis.py` on `allenai/WildChat` (train split).",
        "",
        "---",
        "",
        "## 1. Dataset overview",
        "",
        f"- **Rows (conversations):** {s.get('n_rows', 'N/A')}",
        f"- **Rows with valid timestamp:** {s.get('n_valid_ts', 'N/A')}",
        f"- **Columns:** {s.get('columns', [])}",
        "",
        "---",
        "",
        "## 2. Timestamp validation",
        "",
        f"- **Min timestamp (UTC):** {s.get('ts_min_dt', 'N/A')}",
        f"- **Max timestamp (UTC):** {s.get('ts_max_dt', 'N/A')}",
        f"- **Time span:** {s.get('ts_span_days', 0):.1f} days" if isinstance(s.get('ts_span_days'), (int, float)) else "- **Time span:** N/A",
        f"- **Duplicate timestamps:** {s.get('n_duplicate_ts', 'N/A')}",
        f"- **Strictly increasing when sorted:** {s.get('strictly_increasing', 'N/A')}",
        "",
        "Timestamps are usable for ordering and arrival-process analysis. Hour-of-day and day-of-week histograms show whether load varies by time (see plots).",
        "",
        "---",
        "",
        "## 3. Session length distribution",
        "",
        f"- **Mean / median turns:** {s.get('turn_mean', 0):.2f} / {s.get('turn_median', 0):.0f}" if isinstance(s.get('turn_mean'), (int, float)) else "- N/A",
        f"- **P90 / P95 / P99:** {s.get('turn_p90', 0):.0f} / {s.get('turn_p95', 0):.0f} / {s.get('turn_p99', 0):.0f}" if isinstance(s.get('turn_p90'), (int, float)) else "- N/A",
        f"- **Fraction ≥2 turns:** {(s.get('frac_ge2') or 0)*100:.2f}%",
        f"- **Fraction ≥3 turns:** {(s.get('frac_ge3') or 0)*100:.2f}%",
        f"- **Fraction ≥5 turns:** {(s.get('frac_ge5') or 0)*100:.2f}%",
        f"- **Fraction ≥10 turns:** {(s.get('frac_ge10') or 0)*100:.2f}%",
        f"- **Turn field vs manual count from `conversation`:** correlation {s.get('turn_field_vs_conv_corr') or 0:.4f}, exact agreement {(s.get('turn_field_vs_conv_agree') or 0)*100:.2f}%",
        "",
        "---",
        "",
        "## 4. Arrival process",
        "",
        f"- **Inter-arrival (s):** mean {s.get('ia_mean') or 0:.2f}, median {s.get('ia_median') or 0:.2f}, P95 {s.get('ia_p95') or 0:.2f}",
        f"- **Conversations per hour (mean / std):** {s.get('arrival_rate_per_hour_mean') or 0:.1f} / {s.get('arrival_rate_per_hour_std') or 0:.1f}",
        f"- **Arrival rate varies by hour-of-day (std across hours):** {s.get('arrival_by_hour_std') or 0:.1f}",
        f"- **Arrival rate varies by day-of-week (std):** {s.get('arrival_by_dow_std') or 0:.1f}",
        "",
        "---",
        "",
        "## 5. Model-based concurrency",
        "",
        "Session start time was approximated as `timestamp - turn_count × K` seconds (K = 10 or 30). Each session is active on [start_time, timestamp]. **This is model-based, not observed.**",
        "",
        f"- **K=10s/turn:** peak concurrent sessions {s.get('concurrency_10s_peak') or 0:.0f}, mean {s.get('concurrency_10s_mean') or 0:.1f}",
        f"- **K=30s/turn:** peak concurrent sessions {s.get('concurrency_30s_peak') or 0:.0f}, mean {s.get('concurrency_30s_mean') or 0:.1f}",
        "",
        "---",
        "",
        "## 6. Structural characteristics",
        "",
        f"- **Avg words per turn (proxy for tokens):** {s.get('avg_words_per_turn_overall') or 0:.1f}",
        f"- **Turn count by model:** {s.get('turn_by_model', {})}",
        f"- **Conversation count by model:** {s.get('count_by_model', {})}",
        "",
        "---",
        "",
        "## 7. Implications for scheduling",
        "",
        "- Session length and arrival process support **session-aware workload generators** and **time-varying load** (e.g. by hour or day).",
        "- Model-based concurrency illustrates **plausible concurrent load** under simple duration assumptions; real systems would need calibration.",
        "- No per-turn timestamps: intra-session timing and true session duration are **not observable**; any turn-level arrival process is model-derived.",
        "",
        "---",
        "",
        "## 8. Caveats",
        "",
        "- **No per-message timestamps:** Only the last-turn timestamp per conversation is available. Inter-arrival is between conversation end times, not start times.",
        "- **Concurrency is synthetic:** Based on assumed duration = turn × 10s or 30s.",
        "- **Words per turn** are a proxy for tokens (no tokenizer applied).",
        "",
        "---",
        "",
        "**Artifacts:** Plots and CSVs in `analysis/wildchat/output/`. Stats in `wildchat_empirical_stats.json`.",
    ]
    report_path = out_dir.parent / "wildchat_empirical_v1.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
