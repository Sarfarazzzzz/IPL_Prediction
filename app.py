import streamlit as st
import pandas as pd
import pickle
import numpy as np

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
        return matches_df
    except FileNotFoundError:
        st.error("Matches file 'matches.csv' not found.")
        return None

matches_df = load_match_data()

# Stop the app if any essential file is missing
if not all([model, team_encoder, venue_encoder, matches_df is not None]):
    st.stop()

# --- Prepare UI lists from the loaded artifacts and data ---
# This uses the STRING names from the encoder for the user to see
all_teams = sorted(team_encoder.classes_)
all_venues = sorted(venue_encoder.classes_)
all_cities = sorted(matches_df['city'].dropna().unique())

def predict_probability(state_df):
    """Takes a DataFrame and returns the win probability."""
    # The model expects a NumPy array, so we provide one
    return model.predict_proba(state_df.values)[0][1]

# --- Streamlit UI ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("Simulate a match ball-by-ball and see the win probability change in real-time.")

with st.sidebar:
    st.header("⚙️ Match Setup")
    # The user interacts with the list of STRING names
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
        
        # Store the STRING names for display
        st.session_state.batting_team_name = batting_team
        st.session_state.bowling_team_name = bowling_team
        
        # "Translate" the STRING names into NUMBERS for the model
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

    # This dictionary defines the exact feature set for the model using NUMBERS
    features = {
        'batting_team': st.session_state.batting_team_enc,
        'bowling_team': st.session_state.bowling_team_enc,
        'venue': st.session_state.venue_enc,
        'wickets_left': st.session_state.wickets_left,
        'total_runs_so_far': runs_so_far,
        'runs_left': st.session_state.runs_left,
        'balls_left': balls_left,
        'current_run_rate': current_rr,
        'required_run_rate': required_rr
    }
    
    features_df = pd.DataFrame([features])
    current_prob = predict_probability(features_df)
    
    # Display the result using the saved STRING name
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
