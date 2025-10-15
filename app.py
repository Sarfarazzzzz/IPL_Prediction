import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- Caching Functions ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the .pkl file."""
    try:
        with open("lgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'lgb_model.pkl' not found. Please make sure it's in the same directory.")
        return None

@st.cache_data
def load_data():
    """Loads and cleans the matches.csv data, then caches it."""
    try:
        matches_df = pd.read_csv("matches.csv")
    except FileNotFoundError:
        st.error("Matches file 'matches.csv' not found. Please make sure it's in the same directory.")
        return None

    # --- Data Cleaning and Standardization ---

    # 1. Team Name Standardization
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
    }
    team_cols = ['team1', 'team2', 'toss_winner', 'winner']
    for col in team_cols:
        matches_df[col] = matches_df[col].replace(team_name_mapping)

    # 2. Venue Name Standardization
    venue_mapping = {
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
        'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
    }
    matches_df['venue'] = matches_df['venue'].replace(venue_mapping)

    # 3. City Name Standardization
    city_mapping = {
        'Bangalore': 'Bengaluru'
    }
    matches_df['city'] = matches_df['city'].replace(city_mapping)

    return matches_df

# --- Load Assets ---
model = load_model()
matches_df = load_data()

if model is None or matches_df is None:
    st.stop()

# --- Pre-computation and Mappings ---
all_teams = sorted(matches_df['team1'].dropna().unique())
team_encoding = {team: i for i, team in enumerate(all_teams)}

all_venues = sorted(matches_df['venue'].dropna().unique())
venue_encoding = {venue: i for i, venue in enumerate(all_venues)}

city_to_home_team = {}
matches_df_cleaned = matches_df.dropna(subset=['city', 'team1'])
for index, row in matches_df_cleaned.iterrows():
    if row['city'] not in city_to_home_team:
        city_to_home_team[row['city']] = row['team1']

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="IPL In-Play Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL In-Play Match Winner Prediction")
st.markdown("Predict the **batting team's win probability** as the match progresses ball-by-ball during the second innings.")

# Sidebar inputs
st.sidebar.header("⚙️ Match Settings")

batting_team = st.sidebar.selectbox("Select Batting Team", all_teams)
bowling_team = st.sidebar.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team])

sorted_cities = sorted(matches_df['city'].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", sorted_cities)

possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].unique())
venue = st.sidebar.selectbox("Select Venue", possible_venues)

target_runs = st.sidebar.number_input("Target Runs to Win", min_value=1, max_value=400, value=150)

# ------------------------------
# Prediction Logic
# ------------------------------

if st.sidebar.button("Simulate and Predict"):
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same.")
    else:
        try:
            batting_team_enc = team_encoding[batting_team]
            bowling_team_enc = team_encoding[bowling_team]
            venue_enc = venue_encoding[venue]
        except KeyError as e:
            st.error(f"Encoding error: {e}. One of the selected teams or venue might not be in the training data.")
            st.stop()

        home_team = city_to_home_team.get(selected_city)
        is_home_team = 1 if batting_team == home_team else 0
        
        balls = np.arange(1, 121)
        probabilities = []
        
        wickets_left = 10
        
        for ball in balls:
            balls_so_far = ball
            balls_left = 120 - balls_so_far
            
            runs_so_far = int((balls_so_far / 6) * 8)
            runs_left = target_runs - runs_so_far

            if runs_left <= 0:
                probabilities.append(1.0)
                continue

            current_rr = (runs_so_far * 6 / balls_so_far) if balls_so_far > 0 else 0
            required_rr = (runs_left * 6 / balls_left) if balls_left > 0 else 100
            run_rate_diff = current_rr - required_rr

            features = pd.DataFrame([[
                batting_team_enc, bowling_team_enc, venue_enc,
                balls_so_far, balls_left,
                runs_so_far, runs_left,
                current_rr, required_rr,
                wickets_left, run_rate_diff, is_home_team
            ]], columns=[
                'batting_team', 'bowling_team', 'venue',
                'balls_so_far', 'balls_left',
                'total_runs_so_far', 'runs_left',
                'current_run_rate', 'required_run_rate',
                'wickets_left', 'run_rate_diff', 'is_home_team'
            ])

            prob = model.predict_proba(features)[0][1]
            probabilities.append(prob)

        st.header("Prediction Results")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Win Probability for {batting_team}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(balls / 6, probabilities, label=f"{batting_team} Win Probability", color="#0072B2", linewidth=2)
            ax.axhline(0.5, linestyle="--", color="red", alpha=0.7, label="50% Mark")
            
            ax.fill_between(balls / 6, probabilities, 0.5, where=(np.array(probabilities) >= 0.5), facecolor='green', alpha=0.2)
            ax.fill_between(balls / 6, probabilities, 0.5, where=(np.array(probabilities) < 0.5), facecolor='red', alpha=0.2)
            
            ax.set_xlabel("Overs")
            ax.set_ylabel("Win Probability")
            ax.set_title(f"Win Probability Simulation: {batting_team} vs {bowling_team}", fontsize=14)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 20)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Final Prediction")
            final_prob = probabilities[-1] * 100
            
            st.metric(label=f"**{batting_team}'s Win Probability**", value=f"{final_prob:.2f}%")

            if final_prob > 55:
                st.success(f"**Conclusion:** {batting_team} is in a strong position to win the match.")
            elif final_prob < 45:
                st.error(f"**Conclusion:** {bowling_team} has the upper hand and is likely to win.")
            else:
                st.warning("**Conclusion:** The match is too close to call. It could go either way!")
            
            st.info("This simulation assumes a consistent run rate and no wickets lost.")













