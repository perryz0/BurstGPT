# BurstGPT: Time-Varying Session Depth and KV-Cache Reuse Potential (v2)

**Exploratory memo.** Empirical analysis of conversation depth over time in the BurstGPT trace, with robustness checks and research-grade framing.

---

## 1. Motivation

LLM serving systems can reuse KV-cache across turns within the same conversation, reducing redundant computation. If session depth (number of turns per conversation) varies over time, then **KV-cache reuse potential is time-varying**: periods with more multi-turn sessions offer more reuse opportunity; periods dominated by single-shot queries do not. This memo examines whether the BurstGPT dataset **exhibits evidence of** such time variation. We distinguish two questions: (1) **existence** of multi-turn sessions and of variation in their share over time; (2) **magnitude** of that variation. Conclusions are framed in terms of supporting evidence, not causal proof.

---

## 2. Dataset and Assumptions

- **Source:** `data/BurstGPT_1.csv` (in-repository version). CSV columns: `Timestamp`, `Model`, `Request tokens`, `Response tokens`, `Total tokens`, `Log Type`.
- **Timestamp:** Request submission time in **seconds from 0:00:00 on the first day** (relative time; local time zone per README).
- **Session ID:** The repository README states that Release v1.2 provides a `Session ID` column. The copy in this repository **does not include** it. Sessions were **inferred** by grouping rows with `Log Type == "Conversation log"` by temporal proximity: same session if the gap to the previous request is ≤ a chosen threshold (baseline 30 minutes); each `API log` row is treated as its own single-turn session. This is a **heuristic proxy** for true conversation sessions; all quantitative results depend on this assumption.
- **Proxy for KV-reuse potential:** We use **number of turns per session (#turns)** as the proxy: more turns ⇒ more opportunity for reuse. We do not measure actual cache hit rates or latency.
- **Scope:** 1,429,737 records; ~1.28M inferred sessions (depending on gap); timestamp range ~61 days.

---

## 3. Methodology

1. **Load & normalize:** Load CSV; normalize `Timestamp`; sort by time; infer session IDs (gap threshold as specified).
2. **Session-level stats:** Per session: start/end time, #turns, duration. Aggregations: mean/median/P90/P95 #turns; fraction of sessions with ≥2 turns (and ≥3 turns) for outlier-robust interpretation.
3. **Temporal aggregation:** Hour of day (0–23, from `start_time mod 86400`); 1-hour bins along the trace. For each: mean #turns, fraction ≥2 and ≥3 turns, session count.
4. **Robustness (v2):** (A) Sensitivity to session-gap heuristic (15m, 30m, 60m). (B) Fraction-based metrics (frac ≥2, ≥3 turns) by hour and over time. (C) Sparse-bin filtering: retain only 1-hour bins with ≥100 sessions; re-examine trends.
5. **Visualization:** Baseline plots (see original report); additional plots for sensitivity, fraction metrics, and filtered time series.

---

## 4. Results

### 4.1 Session-level statistics (baseline 30m gap)

| Metric | Value |
|--------|--------|
| Total sessions | ~1,275,643 |
| Mean #turns | ~1.12 |
| Median / P90 / P95 #turns | 1 / 1 / 1 |
| Sessions with 1 turn | ~96.1% |
| Sessions with ≥2 turns | ~3.9% |
| Sessions with ≥3 turns | ~0.68% |

**Existence:** A non-negligible fraction of sessions are multi-turn (≥2 turns), and a smaller fraction are “deep” (≥3 turns). **Magnitude:** The majority of sessions are single-turn; the mean #turns is only slightly above 1, so the *average* reuse potential per session is modest, but the *distribution* has a long tail that drives time-varying behavior when the share of multi-turn sessions changes by time of day or period.

### 4.2 Variation by hour of day

- **Mean #turns:** Minimum ~1.01 (hour 3), maximum ~1.78 (hour 8). Std of hourly means ~0.17 (≈15% coefficient of variation).
- **Fraction ≥2 turns:** Ranges from ~0.2% (hour 4) to ~7.2% (hour 16); hours 9–11 and 15–17 show clearly higher fractions. Fraction ≥3 turns is smaller (e.g. ~1.4% at hour 10, ~1.3% at hour 16) but follows a similar diurnal pattern.
- **Interpretation:** The data are consistent with **identifiable periods** (daytime, especially morning and afternoon) having a higher share of multi-turn sessions than night hours. This supports time-varying reuse potential by hour of day.

### 4.3 Variation over the trace

- Average #turns and fraction of multi-turn sessions vary along the trace when binned in 1-hour windows. Sparse bins (few sessions) can show high variance; see §4.5 for filtering.
- Session count per 1-hour window varies strongly over the 61 days, consistent with known periodicity in BurstGPT.

### 4.4 Distribution of #turns

The distribution is highly right-skewed: most sessions have 1 turn; a long tail extends to many turns. This supports that multi-turn sessions **exist** and that their **share** (rather than only the mean) is the relevant quantity for reuse potential when it varies over time.

### 4.5 Robustness

**A. Sensitivity to session-gap heuristic (15m, 30m, 60m)**  
- Total sessions and fraction of sessions with ≥2 turns are stable: frac ≥2 ≈ 3.94–3.96%, frac ≥3 ≈ 0.67–0.68%, mean #turns ≈ 1.12 for all three gaps.  
- The **diurnal pattern** (higher avg #turns and higher frac ≥2 in daytime hours) **persists** for 15m, 30m, and 60m. Std of mean #turns across hours is 0.21 (15m), 0.17 (30m), 0.13 (60m)—so with a longer gap the hourly variation is somewhat damped but the qualitative pattern holds.  
- **Conclusion:** Key findings are **robust** to the choice of gap in this range; we do not rely on a single threshold.

**B. Fraction-based metrics**  
- Fraction of sessions with ≥2 (and ≥3) turns by hour of day and over time is less sensitive to extreme outliers than mean #turns. Plots show the same diurnal and temporal variation as mean #turns, reinforcing that the variation is not driven solely by a few very long sessions.

**C. Sparse-bin filtering (≥100 sessions per 1-hour bin)**  
- After retaining only bins with ≥100 sessions, 619 of 938 bins remain.  
- **Trends persist:** Average #turns vs time and fraction ≥2 turns vs time (filtered) still show clear variation along the trace; the diurnal and multi-day structure is not an artifact of a few low-count bins.

---

## 5. Evidence Summary

- **Does session depth vary over time?**  
  **Yes.** Average #turns and fraction of multi-turn sessions vary by hour of day (with supporting plots and robustness across gap heuristics and sparse-bin filtering) and along the trace. The **evidence supports** time-varying session depth.

- **Are there identifiable periods with more multi-turn sessions?**  
  **Yes.** Daytime hours (e.g. 7–10, 15–17) show higher average depth and higher fraction ≥2 turns; night hours (e.g. 0–6) are lower. The **evidence supports** identifiable phases where multi-turn sessions are relatively more frequent.

- **Does the data support the claim that KV-reuse potential is time-varying?**  
  **Yes, as a proxy.** We do not measure actual KV reuse. Given that #turns is a direct proxy for reuse opportunity, the observed variation in #turns and in the fraction of multi-turn sessions over time **supports** that KV-cache reuse potential in this workload is time-varying. Magnitude of benefit would depend on system design and trace details.

---

## 6. Discussion

**What is robust across heuristics?**  
The existence of multi-turn sessions, the approximate fraction of sessions with ≥2 turns (~4%), and the **qualitative** diurnal pattern (daytime higher than night) are robust to the 15m/30m/60m session-gap choice and to sparse-bin filtering. The exact hourly values and the std of mean #turns across hours depend somewhat on the gap.

**What remains uncertain without official Session IDs?**  
True conversation boundaries are unknown. Inferred sessions may merge distinct conversations (long gap) or split one conversation (short gap). So the **absolute** level of multi-turn share and the **exact** magnitude of hourly variation could shift with ground-truth session IDs (e.g. from v1.2). The **direction** of variation (daytime vs night, and variation over the trace) is what we treat as supported.

**Why is the evidence still sufficient to motivate adaptive / KV-aware scheduling?**  
We do not need a precise estimate of reuse potential to justify considering it. The data show that (1) multi-turn sessions exist, (2) their share varies by time of day and over the trace, and (3) these patterns are stable under reasonable changes to the session heuristic. That is enough to motivate designs that **adapt** to time-varying session depth (e.g. scheduling or batching that accounts for higher reuse potential in certain phases) rather than assuming a stationary request structure.

---

## 7. Implications for LLM Serving Systems

LLM serving workloads can exhibit **time-varying KV-cache reuse potential** when the mix of single-turn vs multi-turn sessions changes over time, as observed in BurstGPT. This implies **workload phase changes**: the same system may see periods where a larger fraction of requests belong to multi-turn conversations (higher reuse potential) and periods where single-shot queries dominate (lower reuse potential). Schedulers and capacity planners that assume **stationary** request structure (e.g. fixed ratio of prefill-to-decode or fixed average session length) may be suboptimal if they ignore this variation. The findings do not prescribe a particular mechanism (e.g. specific batching or eviction policy) but **motivate** the use of traces with session structure—and time-varying session depth—when evaluating or designing KV-cache-aware and adaptive scheduling strategies.

---

## 8. Artifacts and References

- **Baseline analysis:** `analysis/kv_reuse_analysis.py`; outputs in `analysis/output/` (unchanged).
- **Robustness (v2):** `analysis/kv_reuse_robustness.py`.  
  **New outputs:**  
  - Sensitivity: `by_hour_15m.csv`, `by_hour_30m.csv`, `by_hour_60m.csv`; `by_bin_15m.csv`, `by_bin_30m.csv`, `by_bin_60m.csv`; `sensitivity_session_gap.csv`; plot `sensitivity_avg_turns_by_hour.png`.  
  - Fraction metrics: `fraction_ge2_by_hour.png`, `fraction_ge2_vs_time.png`, `fraction_ge3_by_hour.png`, `fraction_ge3_vs_time.png`.  
  - Sparse-bin: `by_bin_30m_min100.csv`; plots `avg_turns_vs_time_min100.png`, `fraction_ge2_vs_time_min100.png`.
- **Original report:** `burstgpt_kv_reuse_analysis.md`.
