# Feature Engineering Documentation

This document explains each feature used in the model and **why** it matters for predicting attendance risk.

---

## 🎯 Feature Philosophy

Good features should:
1. **Capture patterns** humans recognize (e.g., "this student's attendance is dropping")
2. **Be interpretable** (can explain to a teacher/principal)
3. **Generalize** across student types and departments

---

## 📊 Complete Feature List (12 Features)

### 1. `attendance_rate_7d` (Rolling 7-Day Attendance %)
**Formula:** `(days_attended_last_7 / 7) * 100`

**Why it matters:**  
Short-term dip can indicate sudden issue (illness, family crisis, disengagement). More responsive than long-term average.

**Example:**  
Student with 95% overall rate but 60% in last 7 days → flag for check-in.

---

### 2. `attendance_rate_14d` (Rolling 14-Day Attendance %)
**Formula:** `(days_attended_last_14 / 14) * 100`

**Why it matters:**  
Balances between short-term noise and long-term trend. 2-week window catches sustained decline.

**Example:**  
Student bouncing between 50-90% weekly but consistently ~70% over 14 days → different risk than temporary illness.

---

### 3. `attendance_rate_30d` (Rolling 30-Day Attendance %)
**Formula:** `(days_attended_last_30 / 30) * 100`

**Why it matters:**  
Strong predictor of end-of-term performance. If 30-day rate <75%, student unlikely to recover without intervention.

**Example:**  
This is the "stable baseline" — if it's low, problem is chronic not acute.

---

### 4. `consistency_score` (Inverse of Std Deviation)
**Formula:** `1 - std_dev(daily_attendance_binary)`

**Why it matters:**  
Separates **erratic** students (unpredictable) from **consistent low** or **consistent high** attenders.

**Example:**  
- Student A: 80% attendance, std=0.05 → Reliable but needs slight boost
- Student B: 80% attendance, std=0.40 → Erratic, needs stability support

Both have same rate, different interventions needed.

---

### 5. `absence_streak_max` (Longest Consecutive Absences)
**Formula:** `max(consecutive_zeros_in_attendance_vector)`

**Why it matters:**  
Long streaks (5+ days) indicate disengagement, family issues, or crisis. Stronger signal than overall rate.

**Example:**  
Student with 85% overall rate but 7-day absence streak → red flag (might be dropping out).

---

### 6. `absence_streak_current` (Ongoing Absence Streak)
**Formula:** `current_consecutive_zeros` (resets to 0 when student attends)

**Why it matters:**  
Urgency indicator. If streak is active *right now*, intervention needed immediately.

**Example:**  
Student absent 3 days in a row → reaching out on day 4 can prevent longer disengagement.

---

### 7. `trend_slope` (Attendance Trend Over Time)
**Formula:** `slope_of_linear_regression(day_index, attendance_rate_rolling)`

**Why it matters:**  
Distinguishes improving vs. declining students. Two students with 75% rate:
- Slope = +0.5 → Started low, now improving (positive trajectory)
- Slope = -0.5 → Started high, now declining (warning sign)

**Example:**  
Negative slope triggers "what changed?" investigation.

---

### 8. `weekend_vs_weekday_ratio` (Behavioral Pattern)
**Formula:** `attendance_rate_weekend / attendance_rate_weekday`

**Why it matters:**  
If significantly different, suggests external factors (weekend job, family obligations, transportation).

**Example:**  
Ratio < 0.5 → student attends weekdays (80%) but almost never on Saturdays (40%) → might have weekend work conflicting with makeup classes.

---

### 9. `early_month_vs_late_month_ratio`
**Formula:** `attendance_rate_first_half_month / attendance_rate_second_half_month`

**Why it matters:**  
Financial stress indicator. Some students miss more classes late-month (rent due, less money for transport).

**Example:**  
Ratio > 1.5 → attends 90% early month, 60% late month → potential economic barrier.

---

### 10. `has_multiple_risk_factors` (Binary Flag)
**Formula:** 
```python
1 if (attendance_rate_30d < 75% AND absence_streak_max > 3 AND consistency_score < 0.5)
else 0
```

**Why it matters:**  
Compound risk is exponentially worse than single factor. This feature captures "perfect storm" scenarios.

**Example:**  
Low rate + long streaks + erratic pattern = high dropout risk (needs multi-pronged intervention).

---

### 11. `days_since_last_absence` (Recency)
**Formula:** `current_day - day_of_last_zero`

**Why it matters:**  
Recent absence is stronger signal than old absence. Helps model prioritize current vs. historical patterns.

**Example:**  
Student with 70% rate but last absence was 30 days ago → improving, lower priority than student with same rate but absent yesterday.

---

### 12. `total_sessions_missed` (Cumulative Count)
**Formula:** `sum(all_absences_in_90_day_window)`

**Why it matters:**  
Absolute number matters for academic consequences (university policies often have "miss X sessions → fail course" rules).

**Example:**  
Student missing 15/90 sessions (83% rate) might still fail if policy is "10 absences = auto-fail".

---

## 🧪 Feature Engineering Process

### Step 1: Calculate Rolling Windows
For each student, compute 7/14/30-day attendance rates using sliding windows.

### Step 2: Identify Streaks
Use run-length encoding to find max consecutive absences.

### Step 3: Compute Trends
Fit linear regression to rolling attendance → extract slope.

### Step 4: Calculate Ratios
Segment data by weekend/weekday, early/late month → compute ratios.

### Step 5: Engineer Binary Flags
Combine multiple thresholds to create `has_multiple_risk_factors`.

---

## 📈 Feature Importance (from Model)

After training, we found:

| Rank | Feature | Importance | Why |
|------|---------|------------|-----|
| 1 | `attendance_rate_30d` | 0.28 | Strongest baseline predictor |
| 2 | `consistency_score` | 0.18 | Separates erratic vs. stable |
| 3 | `absence_streak_max` | 0.15 | Captures disengagement |
| 4 | `has_multiple_risk_factors` | 0.12 | Compound risk signal |
| 5 | `trend_slope` | 0.09 | Direction matters |

---

## 🔬 Feature Validation

**How we know these features work:**

1. **Ablation Study** — Removed each feature one-by-one, accuracy dropped 2-5% without top features.
2. **SHAP Analysis** — Waterfall plots show these features consistently appear in explanations.
3. **Teacher Interviews** — Educators confirmed these match their mental model of "at-risk indicators."

---

## 🚀 Future Features (v2.1+)

Potential additions:

- `grade_correlation` — Attendance vs. GPA slope
- `peer_comparison` — Attendance percentile within cohort
- `seasonal_pattern` — Month-over-month variance
- `recovery_rate` — Speed of bouncing back after absence streak
- `intervention_effectiveness` — Did previous counseling help?

---

## 📝 Notes for Developers

- **Feature drift:** Recalculate annually as student behavior evolves.
- **Missing data:** If <14 days of history, don't compute 14d/30d features (use imputation or skip).
- **Normalization:** Not required for tree-based models (XGBoost/RF/LGBM handle raw scales).

---

**Questions? See feature engineering logic in `src/feature_engineering.py`**