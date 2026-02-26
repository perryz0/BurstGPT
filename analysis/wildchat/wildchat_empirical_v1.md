# WildChat Empirical Workload Analysis (v1)

Evidence-based summary from running `run_empirical_analysis.py` on `allenai/WildChat` (train split).

---

## 1. Dataset overview

- **Rows (conversations):** 529428
- **Rows with valid timestamp:** 529428
- **Columns:** ['conversation_id', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted']

---

## 2. Timestamp validation

- **Min timestamp (UTC):** 2023-04-10 00:01:08+00:00
- **Max timestamp (UTC):** 2023-11-09 16:22:27+00:00
- **Time span:** 213.7 days
- **Duplicate timestamps:** 12219
- **Strictly increasing when sorted:** True

Timestamps are usable for ordering and arrival-process analysis. Hour-of-day and day-of-week histograms show whether load varies by time (see plots).

---

## 3. Session length distribution

- **Mean / median turns:** 2.35 / 1
- **P90 / P95 / P99:** 5 / 7 / 13
- **Fraction ≥2 turns:** 42.11%
- **Fraction ≥3 turns:** 26.19%
- **Fraction ≥5 turns:** 12.43%
- **Fraction ≥10 turns:** 2.73%
- **Turn field vs manual count from `conversation`:** correlation 1.0000, exact agreement 100.00%

---

## 4. Arrival process

- **Inter-arrival (s):** mean 35.28, median 22.00, P95 108.00
- **Conversations per hour (mean / std):** 105.0 / 61.0
- **Arrival rate varies by hour-of-day (std across hours):** 3860.1
- **Arrival rate varies by day-of-week (std):** 6390.4

---

## 5. Model-based concurrency

Session start time was approximated as `timestamp - turn_count × K` seconds (K = 10 or 30). Each session is active on [start_time, timestamp]. **This is model-based, not observed.**

- **K=10s/turn:** peak concurrent sessions 14, mean 1.4
- **K=30s/turn:** peak concurrent sessions 26, mean 3.1

---

## 6. Structural characteristics

- **Avg words per turn (proxy for tokens):** 168.4
- **Turn count by model:** {'gpt-3.5-turbo': 2.4139892599379547, 'gpt-4': 1.806717909830415, 'gpt-4-1106-preview': 2.5142231947483586}
- **Conversation count by model:** {'gpt-3.5-turbo': 470947, 'gpt-4': 58024, 'gpt-4-1106-preview': 457}

---

## 7. Implications for scheduling

- Session length and arrival process support **session-aware workload generators** and **time-varying load** (e.g. by hour or day).
- Model-based concurrency illustrates **plausible concurrent load** under simple duration assumptions; real systems would need calibration.
- No per-turn timestamps: intra-session timing and true session duration are **not observable**; any turn-level arrival process is model-derived.

---

## 8. Caveats

- **No per-message timestamps:** Only the last-turn timestamp per conversation is available. Inter-arrival is between conversation end times, not start times.
- **Concurrency is synthetic:** Based on assumed duration = turn × 10s or 30s.
- **Words per turn** are a proxy for tokens (no tokenizer applied).

---

## 9. Summary and assessment

**Summary of empirical findings**
- **529,428** conversations over **~214 days** (Apr–Nov 2023). All rows have valid UTC timestamps.
- **Session length:** Mean 2.35 turns, median 1; **42%** of sessions have ≥2 turns, **26%** ≥3, **12%** ≥5. The `turn` field matches manual count from `conversation` (100% agreement).
- **Arrival process:** Inter-arrival mean 35 s, median 22 s, P95 108 s. ~105 conversations/hour on average; rate varies by hour-of-day and day-of-week.
- **Model-based concurrency (10s or 30s per turn):** Peak 14–26 concurrent sessions; mean 1.4–3.1. Illustrative only.
- **By model:** gpt-3.5-turbo dominates (471k conv); gpt-4 has shorter mean turn count (1.8 vs 2.4). Avg ~168 words/turn (proxy for tokens).

**Key plots** (in `analysis/wildchat/output/`)
- `wildchat_session_length_hist.png`, `wildchat_session_length_ccdf.png` — session length distribution
- `wildchat_arrival_rate.png` — conversations per hour over time
- `wildchat_interarrival_hist.png` — inter-arrival distribution
- `wildchat_concurrency_10s.png`, `wildchat_concurrency_30s.png` — model-based concurrency
- `wildchat_turn_distribution_by_model.png` — turn distribution by model
- `wildchat_hour_hist.png`, `wildchat_dow_hist.png`, `wildchat_conv_per_day.png` — time-of-day and daily volume

**Surprises / anomalies**
- **12,219 duplicate timestamps** (same second); ordering is still well-defined (non-decreasing). No impact on arrival-process stats.
- **Strictly increasing when sorted** is True in the sense that deltas are ≥ 0 (duplicates allowed).
- Session length is more multi-turn than a “single query” stereotype: 42% ≥2 turns supports session-aware modeling.

**Assessment: Is WildChat usable for session-aware workload modeling?**  
**Yes.** We can reliably compute session-length distributions, conversation-level arrival process, and time-of-day/day-of-week patterns. The `turn` field is consistent with the `conversation` list. Limitations: no per-turn timestamps (so no observed intra-session gaps or true session duration), and concurrency is model-based. For building session-aware workload generators and studying time-varying load, WildChat is suitable; for turn-level arrival or exact session duration, additional assumptions are required.

---

**Artifacts:** Plots and CSVs in `analysis/wildchat/output/`. Stats in `wildchat_empirical_stats.json`.