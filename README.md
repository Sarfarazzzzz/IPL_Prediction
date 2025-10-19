# 🏏 IPL Live Match Win Predictor

A Streamlit web application that predicts the live win probability for the chasing team during an Indian Premier League (IPL) T20 match. The prediction updates ball-by-ball based on the current match state, utilizing an XGBoost model trained on historical data.

---

## ✨ Features

* **Live Prediction:** Calculates win probability after every ball using a trained XGBoost model.
* **Match Setup:** Select batting team, bowling team, venue (auto-filtered by city), and target score.
* **Ball-by-Ball Input:** Enter runs scored and wicket status for each delivery.
* **Probability Chart:** Visualizes the win probability trend for both teams throughout the chase using Matplotlib.
* **State Jump:** Allows setting a custom match state (overs bowled, runs left, wickets left) to analyze specific scenarios.
* **Stable Start:** Uses a target-sensitive heuristic for the initial probability (Ball 0) and smoothly blends into the ML model's predictions over the first two overs (12 balls) to manage potential early-game volatility inherent in the model.
* **Data Cleaning:** Includes comprehensive mapping for team names and venue variations to ensure consistency.

---

## ⚙️ How it Works

1.  **Initial Probability (Ball 0):** A starting win probability is calculated based on the target score relative to a par score (175).
2.  **Blend-in Phase (Overs 1-2 / Balls 1-12):** To ensure a smooth start during the model's most volatile phase, the app transitions from the initial probability to the machine learning model's prediction over the first 12 balls. The weight given to the model's prediction increases with each ball. Predictions during this phase use the **original features** the model was trained on, including `wicket_pressure` and `danger_index`.
3.  **ML Model Prediction (Over 3+ / Ball 13+):** From the 13th ball onwards, the XGBoost machine learning model (`xgb_model.pkl`) predicts the win probability based on the current game state, using the **original features** it was trained with.
4.  **Safety Nets:** Hard rules immediately return 0% or 100% for clear game-over scenarios (target reached, wickets lost, balls finished, impossibly high required rate).
5.  **Confidence Adjustment:** A multiplier slightly reduces the predicted probability when very few wickets (1-3) are remaining.

---

## 📊 Data Files

* **Source**: [Kaggle IPL Complete Dataset (2008–2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
* **Files Used**:

    * **`matches.csv`:** Contains match-level information used for data cleaning (venue/team name mapping) and identifying home teams within the app.
  
    * **`deliveries.csv`:** Contains historical ball-by-ball data used for model training (as shown in the notebook). Not directly loaded by the live Streamlit app.

---

## 🤖 Model File

* **`xgb_model.pkl`:** A pre-trained XGBoost Classifier model. It was trained on historical IPL ball-by-ball data.
    * **Features Used During Training (and required for prediction):** `batting_team`, `bowling_team`, `venue`, `balls_so_far`, `balls_left`, `total_runs_so_far`, `runs_left`, `current_run_rate`, `required_run_rate`, `wickets_left`, `run_rate_diff`, `is_home_team`, `phase_Middle`, `phase_Death`, `wicket_pressure`, `danger_index`.

---

## 🛠️ Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure necessary files** are in the repository root:
    * `app.py`
    * `matches.csv`
    * `xgb_model.pkl`
    * `requirements.txt`
    * (Optional: `deliveries.csv` and `ipl_inplay_prediction (3).ipynb` for reference/retraining)
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
5.  **Use the app:** Open the provided URL. Set up the match in the sidebar, click "Start / Reset Simulation", and input ball-by-ball details.

---

## 🌱 Future Work & Potential Improvements

* **Model Sensitivity:** The current model, trained with `wicket_pressure` and `danger_index`, shows some volatility in the early overs (even with the blend-in). Future work could involve:
    * Retraining the model **without** these engineered features to see if stability improves.
    * Experimenting with different feature engineering or robust scaling techniques (applied consistently during training *and* prediction).
    * Incorporating more recent IPL seasons into the training data.
* **Player-Specific Features:** Enhance the model by adding features related to the specific batsmen at the crease and the bowler.
* **UI Enhancements:** Add more visualizations, potentially showing the impact of the next possible ball outcomes.
* **Contextual Factors:** Consider incorporating factors like pitch conditions or weather.

---

