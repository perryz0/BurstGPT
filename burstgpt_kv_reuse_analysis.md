# BurstGPT: Time-Varying Session Depth and KV-Cache Reuse Potential

**Report:** Empirical analysis of conversation depth over time in the BurstGPT trace.

---

## 1. Motivation

LLM serving systems can reuse KV-cache across turns within the same conversation, reducing redundant computation. If session depth (number of turns per conversation) varies over time, then **KV-cache reuse potential is time-varying**: periods with more multi-turn sessions offer more reuse; periods dominated by single-shot queries do not. This report tests whether the BurstGPT dataset exhibits such time variation using timestamps and session structure.

---

## 2. Dataset Description

- **Source:** `data/BurstGPT_1.csv` (in-repository version).
- **Format:** CSV with columns: `Timestamp`, `Model`, `Request tokens`, `Response tokens`, `Total tokens`, `Log Type`.
- **Timestamp:** Request submission time in **seconds from 0:00:00 on the first day** (relative time; local time zone per README).
- **Session ID:** The repository README states that Release v1.2 provides a `Session ID` column (conversation ID; same value = same conversation). The copy of `BurstGPT_1.csv` in this repository **does not include** `Session ID` or `Elapsed time`. Sessions were **inferred** for this analysis as follows: rows with `Log Type == "Conversation log"` were grouped by temporal proximity (same session if the gap to the previous request is ≤ 30 minutes); each `API log` row was treated as its own single-turn session. This is a heuristic proxy for true conversation sessions.
- **Scope:** 1,429,737 records; inferred 1,275,643 sessions; timestamp range 5–5,269,973 s (~61 days).

---

## 3. Methodology

1. **Load & normalize:** Load CSV with pandas; normalize `Timestamp` to numeric; sort by time. Infer session IDs when missing (as above).
2. **Session-level stats:** For each session: start time, end time, number of turns (# records), duration. **#turns is used as the proxy for KV-reuse potential** (more turns ⇒ more reuse opportunity).
3. **Temporal aggregation:**
   - **Hour of day:** `hour = (start_time mod 86400) / 3600` (integer 0–23). Aggregate mean, P90, P95 of #turns and session count per hour.
   - **Time windows:** 1-hour bins over the trace (bin = floor(start_time / 3600) * 3600). Same aggregates per bin.
4. **Visualization:** (1) Average #turns vs time (days); (2) Session count vs time; (3) Histogram of #turns per session; (4) Box plot of #turns by hour of day; (5) Bar plot of average #turns by hour of day.
5. **Evidence:** Inspect whether average #turns and session depth distribution vary by hour and over the trace.

---

## 4. Results

### 4.1 Session-level statistics

| Metric            | Value   |
|-------------------|--------|
| Total sessions    | 1,275,643 |
| Mean #turns       | 1.12   |
| Median #turns     | 1      |
| P90 #turns        | 1      |
| P95 #turns        | 1      |
| Sessions with 1 turn | 1,225,315 (96.1%) |
| Sessions with 2+ turns | 50,328 (3.9%) |

Most sessions are single-turn; a small fraction are multi-turn and drive KV-reuse potential.

### 4.2 Variation by hour of day

Average #turns per session by hour (0–23):

- **Minimum:** ~1.01 (hour 3).
- **Maximum:** ~1.78 (hour 8).
- **Std of hourly averages:** ~0.17 (mean ~1.12 ⇒ ~15% coefficient of variation across hours).

Hours 7–10 and 15–17 show consistently higher average depth (e.g., hour 8: 1.78, hour 10: 1.27, hour 16: 1.18). Night hours (0–6) are slightly lower. This indicates **identifiable periods with more multi-turn sessions** (daytime, especially morning).

### 4.3 Variation over the trace

- **Average #turns vs time (1-hour bins):** The line plot shows variation along the trace; some intervals have higher average #turns due to local mix of single- vs multi-turn sessions. Sparse bins (few sessions) can show high variance.
- **Session count vs time:** Session count per 1-hour window varies strongly over the 61 days (workload intensity is time-varying), which is consistent with the known periodicity reported in the BurstGPT paper.

### 4.4 Distribution of #turns

The histogram of #turns per session is highly right-skewed: most sessions have 1 turn; a long tail has 2, 3, … up to large values. This supports that **multi-turn sessions exist and contribute to time-varying reuse potential** when their share varies by time of day or by period in the trace.

---

## 5. Evidence Summary

- **Does session depth vary over time?**  
  **Yes.** Average #turns per session varies by hour of day (std of hourly means ~0.17; range ~1.01–1.78). It also varies along the trace when binned in 1-hour windows.

- **Are there identifiable periods with more multi-turn sessions?**  
  **Yes.** Hours 7–10 and 15–17 (daytime) show higher average #turns; night hours (e.g., 0–6) are slightly lower. So there are identifiable times when multi-turn sessions are relatively more frequent.

- **Does the data support the claim that KV-reuse potential is time-varying?**  
  **Yes.** #turns is a direct proxy for reuse opportunity; we observe that (1) #turns varies across sessions (long tail of multi-turn sessions) and (2) the average #turns varies by hour of day and along the trace. Therefore the data support that **KV-cache reuse potential is time-varying** in this workload.

**Caveat:** Session boundaries were inferred (30-minute gap, Conversation log only) because the in-repo CSV lacks the official `Session ID` column. Results would be refined with the v1.2 trace that includes true session IDs.

---

## 6. Implications for LLM Serving & KV-Cache-Aware Scheduling

- **Scheduling:** Systems that exploit KV-cache reuse can expect **higher benefit during daytime hours** (e.g., 7–10, 15–17) when average session depth is higher, and lower benefit during night hours when single-turn sessions dominate.
- **Capacity planning:** Time-varying session depth implies time-varying efficiency gains from reuse; peak reuse potential need not align with peak request rate (session count), so planning should consider both.
- **Trace-driven evaluation:** Replay or synthetic workloads based on BurstGPT should preserve or resample **session depth by time of day** (and, if available, true session IDs from v1.2) to reflect realistic KV-reuse potential over time.

---

*Analysis script: `analysis/kv_reuse_analysis.py`.*

**Generated outputs:**
- Plots: `analysis/output/avg_turns_vs_time.png`, `analysis/output/session_count_vs_time.png`, `analysis/output/turns_histogram.png`, `analysis/output/turns_vs_hour_of_day.png`, `analysis/output/avg_turns_by_hour.png`
- Data: `analysis/output/session_stats.csv`, `analysis/output/by_hour.csv`, `analysis/output/by_time_bin.csv`
