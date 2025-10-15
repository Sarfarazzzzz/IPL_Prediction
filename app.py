import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Load All Model Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the model and the fitted encoders from disk."""
    try:
        with open("lgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("team_encoder.pkl", "rb") as f:
            team_encoder = pickle.load(f)
        with open("venue_encoder.pkl", "rb") as f:
            venue_encoder = pickle.load(f)
        return model, team_encoder, venue_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Please ensure lgb_model.pkl, team_encoder.pkl, and venue_encoder.pkl are in your GitHub repository.")
        return None, None, None

model, team_encoder, venue_encoder = load_artifacts()

@st.cache_data
def load_match_data():
    """Loads the matches.csv for UI elements."""
    try:
        matches_df = pd.read_csv("matches.csv")
        # Standardize names for a cleaner UI
        team_map = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings', 'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants', 'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'}
        matches_df['team1'] = matches_df['team1'].replace(team_map)
        matches_df['team2'] = matches_df['team2'].replace(team_map)
        matches_df['city'] = matches_df['city'].replace({'Bangalore': 'Bengaluru'})
        return matches_df
    except FileNotFoundError:
        st.error("Matches file 'matches.csv' not found.")
        return None

matches_df = load_match_data()

# Stop the app if any essential file is missing
if not all([model, team_encoder, venue_encoder, matches_df is not None]):
    st.stop()

# --- Prepare UI lists from the loaded artifacts and data ---
all_teams = sorted(team_encoder.classes_)
all_venues = sorted(venue_encoder.classes_)
all_cities = sorted(matches_df['city'].dropna().unique())
city_to_home_team = {row['city']: row['team1'] for _, row in matches_df.dropna(subset=['city', 'team1']).iterrows()}

def predict_probability(state_df):
    """Takes a DataFrame and returns the win probability."""
    return model.predict_proba(state_df.values)[0][1]

# --- Streamlit UI ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("Simulate a match ball-by-ball or jump to any point in the chase.")

with st.sidebar:
    st.header("⚙️ Match Setup")
    batting_team = st.selectbox("Select Batting Team", all_teams)
    bowling_team = st.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team])
    selected_city = st.selectbox("Select City", all_cities)
    possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].unique())
    venue = st.selectbox("Select Venue", possible_venues if possible_venues else all_venues)
    target = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180)

    if st.button("Start / Reset Simulation", type="primary"):
        st.session_state.clear()
        st.session_state.simulation_started = True
        st.session_state.target = target
        st.session_state.runs_left = target
        st.session_state.wickets_left = 10
        st.session_state.balls_so_far = 0
        st.session_state.batting_team_name = batting_team
        st.session_state.bowling_team_name = bowling_team
        st.session_state.is_home_team = 1 if batting_team == city_to_home_team.get(selected_city) else 0
        # Use the loaded encoders to get the correct numerical ID
        st.session_state.batting_team_enc = int(team_encoder.transform([batting_team])[0])
        st.session_state.bowling_team_enc = int(team_encoder.transform([bowling_team])[0])
        st.session_state.venue_enc = int(venue_encoder.transform([venue])[0])


if 'simulation_started' in st.session_state:
    st.header("Current Match State")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", st.session_state.target)
    col2.metric("Runs Left", st.session_state.runs_left)
    col3.metric("Wickets Left", st.session_state.wickets_left)
    col4.metric("Balls Left", 120 - st.session_state.balls_so_far)

    runs_so_far = st.session_state.target - st.session_state.runs_left
    balls_left = 120 - st.session_state.balls_so_far
    current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
    required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 100

    # This dictionary defines the exact feature set for the model
    features = {
        'batting_team': st.session_state.batting_team_enc,
        'bowling_team': st.session_state.bowling_team_enc,
        'venue': st.session_state.venue_enc,
        'balls_so_far': st.session_state.balls_so_far,
        'balls_left': balls_left,
        'total_runs_so_far': runs_so_far,
        'runs_left': st.session_state.runs_left,
        'current_run_rate': current_rr,
        'required_run_rate': required_rr,
        'wickets_left': st.session_state.wickets_left,
        'run_rate_diff': current_rr - required_rr,
        'is_home_team': st.session_state.is_home_team,
    }
    
    features_df = pd.DataFrame([features])
    current_prob = predict_probability(features_df)
    st.metric(label=f"**Current Win Probability for {st.session_state.batting_team_name}**", value=f"{current_prob * 100:.2f}%")

    st.divider()
    st.header("Ball-by-Ball Input")
    runs_scored = st.selectbox("Runs on this ball:", (0, 1, 2, 3, 4, 6))
    is_wicket = st.checkbox("Wicket on this ball?")

    if st.button("Next Ball", type="secondary"):
        if st.session_state.wickets_left <= 0 or st.session_state.runs_left <= 0 or st.session_state.balls_so_far >= 120:
            st.warning("Match is over! Please reset the simulation to start a new one.")
        else:
            st.session_state.runs_left -= runs_scored
            st.session_state.balls_so_far += 1
            if is_wicket:
                st.session_state.wickets_left -= 1
            st.rerun()
else:
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")
