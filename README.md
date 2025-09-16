# 🏏 IPL Match Winner Prediction – Ball-by-Ball In-Play Model

This project builds a real-time machine learning pipeline to predict the **winner of an IPL match after every ball**, based on live match state. The model learns from historical IPL data (2008–2020) to estimate the **batting team's probability of winning**, using only features available up to that point in the match.

---

## 📦 Dataset

- **Source**: [Kaggle IPL Complete Dataset (2008–2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Files used:
  - `matches.csv` – match-level metadata
  - `deliveries.csv` – ball-by-ball match data

---

## 🚀 Features Used

| Type           | Feature Name                  | Description |
|----------------|-------------------------------|-------------|
| **Match Info** | `venue`, `is_home_team`        | Home advantage |
| **Score State**| `total_runs_so_far`, `runs_left`, `wickets_left` | Current game status |
| **Progression**| `balls_so_far`, `balls_left`   | Overs completed / remaining |
| **Performance**| `current_run_rate`, `required_run_rate`, `run_rate_diff` | Pressure metrics |
| **Teams**      | `batting_team`, `bowling_team` | Team IDs (encoded) |

> All features are engineered to avoid any form of data leakage and are derived using only past and present data at each ball.

---

## 🧠 Models Trained & Evaluated

| Model         | Accuracy | Precision (Win) | Recall (Win) | F1 Score (Win) |
|---------------|----------|------------------|---------------|----------------|
| 🎯 LightGBM    | **84.4%** | **84%**           | **86%**        | **85%**         |
| 🚀 XGBoost     | 83.7%     | 83%               | 86%            | 84%             |
| 🌲 Random Forest | 81.0%     | 81%               | 83%            | 82%             |

---
