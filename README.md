# 🏏 IPL Match Winner Prediction – Ball-by-Ball In-Play Model

This project builds a real-time machine learning pipeline to predict the **outcome of an IPL match after every ball**, using historical data from the Indian Premier League (2008–2020).

The model uses match dynamics like current score, overs left, wickets remaining, and venue to estimate the **win probability for the batting team**.

---

## 📂 Dataset

- Source: [Kaggle IPL Complete Dataset (2008–2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Files used:
  - `matches.csv`: match-level information
  - `deliveries.csv`: ball-by-ball data

---

## 🚀 Features Engineered

| Category | Feature |
|----------|---------|
| Match Context | `venue`, `is_home_team`, `target_runs` |
| Game Progress | `balls_so_far`, `balls_left`, `wickets_left` |
| Scoring Dynamics | `total_runs_so_far`, `runs_left`, `current_run_rate`, `required_run_rate`, `run_rate_diff` |
| Outcome | `batting_team_won` (label) |

---

## 🤖 Models Trained

- ✅ **Random Forest Classifier**
- ✅ **XGBoost Classifier**
- ✅ **LightGBM Classifier**

Each model outputs:
- Match outcome prediction (win/lose for batting team)
- Probability score for winning

---

## 📈 Performance Metrics (on sample data)

| Model | Accuracy | Precision (Win) | Recall (Win) | F1 Score (Win) |
|-------|----------|------------------|--------------|----------------|
| Random Forest | 77% | 84% | 76% | 80% |
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |

> ⚠️ Full evaluation is done using a stratified sample of 5,000 records for speed and comparison. Group-based CV is used to prevent match leakage.

---
