# WildChat Windowed Phase Analysis

Targeted analysis: do **average number of turns** and **average context length** change significantly across hourly windows, and do similar trends repeat across days?

---

## What was computed

- **Windowing:** Hour-level windows from conversation timestamp (UTC). Each conversation assigned to the hour bin containing its timestamp. **Context length** = total word count across all utterances in the conversation (no tokenizer).
- **Two levels:** (A) Global hourly bins across the dataset → per-window `avg_turn`, `avg_context_length`, `session_count`. (B) Hour-of-day (0–23) → mean and std of those metrics across all hourly windows that fall in each hour of day.
- **Daily consistency:** For each (date, hour_of_day) we computed avg_turn and avg_context_length. Heatmaps: day × hour_of_day. Pairwise correlation between each day’s 24-dimensional hourly curve (to test if diurnal pattern is consistent across days).

---

## Global hourly variation

- **Number of hourly windows:** 5042
- **Avg turns per window:** mean = 2.339, std = 0.572, P10 = 1.678, P90 = 3.070
- **Avg context length per window:** mean = 586 words, std = 153, P10 = 419, P90 = 766
- **Coefficient of variation (CV):** CV_turn = 0.2447, CV_context = 0.2608

CV quantifies relative variation across hourly windows: CV > 0.1–0.2 suggests non-negligible window-level variation.

---

## Hour-of-day aggregation

Mean and std of avg_turn and avg_context_length for each hour of day (0–23 UTC), aggregated across all days. See plots `wildchat_hour_of_day_avg_turns.png` and `wildchat_hour_of_day_avg_context.png`. If metrics change significantly by hour of day, we expect visible bars and error bars varying across hours.

---

## Daily consistency analysis

- **Days in dataset:** 213
- **Pairwise correlation between daily hourly curves (avg_turn):** mean = 0.0196, std = 0.2608
- **Pairwise correlation between daily hourly curves (avg_context_length):** mean = 0.2131, std = 0.2844

**Interpretation.** Low pairwise correlation does not imply “no pattern.” It means the **exact 24-hour shape** (which hours are high vs low) is **not stable** from day to day. Variation may be driven by **workload-mix heterogeneity** (e.g. different user populations or task mixes on different days) rather than a fixed diurnal periodicity. The heatmaps and hour-of-day bars still show that metrics vary by time of day on average; the decomposition below separates within-day structure from day-to-day stability.

---

## Intra-day vs Inter-day Decomposition

Variance is split into:

- **Intra-day:** For each day, CV of `avg_turn` (and `avg_context_length`) across that day’s 24 hourly bins. This measures how strong **within-day phase structure** is.
- **Inter-day:** For each hour-of-day (0–23), CV of `avg_turn` (and `avg_context_length`) **across days**. The mean of these 24 CVs measures how much that hour **varies from day to day** (stability of the diurnal pattern).

**Computed from** `wildchat_hourly_windows.csv` (see `compute_variance_decomposition.py`). Metrics in `wildchat_windowed_variance_stats.json`.

| Metric | Mean intraday CV | Std intraday CV | Mean interday CV |
|--------|------------------|-----------------|------------------|
| avg_turn | 0.172 | 0.067 | 0.241 |
| avg_context_length | 0.220 | 0.074 | 0.234 |

- **Intraday CV** = mean of (per-day CV across 24 hours). **Interday CV** = mean of (per–hour-of-day CV across days). Days with fewer than 6 hourly bins excluded from intraday (212 days used).

---

## What drives variation?

| Source | CV_turn | CV_context |
|--------|---------|------------|
| **Global** (all hourly windows) | 0.245 | 0.261 |
| **Mean intraday** (within-day structure) | 0.172 | 0.220 |
| **Mean interday** (across-day stability) | 0.241 | 0.234 |

**Interpretation:**

- **Interday CV ≈ global CV** and **interday CV ≥ mean intraday CV** for both metrics. Day-to-day heterogeneity in workload structure is substantial: a given hour of day does not have a stable level of avg_turn or avg_context across days.
- Intraday CV is non-negligible (0.17–0.22), so **within each day** there is meaningful hour-to-hour variation. The mix of effects is **mixed**: both within-day phase structure and across-day heterogeneity contribute; neither clearly dominates to the exclusion of the other, but **interday variation is at least as large as intraday** for this dataset.

---

## Quantitative measures (CV values)

- **CV_turn:** 0.2447
- **CV_context:** 0.2608

---

## Does WildChat show time-phase workload structure that is stable and exploitable for scheduling?

**Window-level variation:** Yes. Avg turns and avg context length vary meaningfully across hourly windows (global CV_turn ≈ 0.25, CV_context ≈ 0.26). Time-phase structure exists at the window level.

**Stability across days:** No. Intra-day vs inter-day decomposition shows **interday CV ≥ mean intraday CV** (turns: 0.24 vs 0.17; context: 0.23 vs 0.22). Pairwise correlation of daily hourly curves is low (turns ~0.02, context ~0.21). So the **diurnal shape is not stable** from day to day; variation is **mixed** (within-day structure plus substantial day-to-day heterogeneity). A fixed “hour-of-day” policy would not capture a consistent pattern.

**Conclusion.** WildChat exhibits **significant window-level phase shifts** (metrics change across hours) but **not** a **stable, repeatable diurnal structure** that is clearly exploitable for scheduling. Exploitation would require either day-specific or adaptive estimation rather than a single global hour-of-day profile.

---

**Artifacts:** `wildchat_hourly_avg_turns.png`, `wildchat_hourly_avg_context.png`, `wildchat_hour_of_day_avg_turns.png`, `wildchat_hour_of_day_avg_context.png`, `wildchat_turn_heatmap.png`, `wildchat_context_heatmap.png`; `wildchat_hourly_windows.csv`, `wildchat_hour_of_day.csv`, `wildchat_windowed_variance_stats.json` in `analysis/wildchat/output/`. Variance decomposition: `analysis/wildchat/compute_variance_decomposition.py`.