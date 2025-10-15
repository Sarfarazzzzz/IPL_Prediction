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

def predict_probability(features_df):
    """Takes a DataFrame with the correct features and returns the win probability."""
    # The model expects a NumPy array, so we provide one
    return model.predict_proba(features_df.values)[0][1]

# --- Streamlit UI ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Win Predictor")

with st.sidebar:
    st.header("⚙️ Match Setup")
    # The user interacts with the list of STRING names
    batting_team_name = st.selectbox("Select Batting Team", all_teams)
    bowling_team_name = st.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team_name])
    selected_city = st.selectbox("Select City", all_cities)
    possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].unique())
    venue_name = st.selectbox("Select Venue", possible_venues if possible_venues else all_venues)
    target = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180)

    if st.button("Start / Reset Simulation", type="primary"):
        st.session_state.clear()
        st.session_state.simulation_started = True
        st.session_state.target = target
        st.session_state.runs_left = target
        st.session_state.wickets_left = 10
        st.session_state.balls_so_far = 0
        st.session_state.batting_team_name = batting_team_name
        
        # "Translate" the STRING names into NUMBERS for the model
        st.session_state.batting_team_enc = int(team_encoder.transform([batting_team_name])[0])
        st.session_state.bowling_team_enc = int(team_encoder.transform([bowling_team_name])[0])
        st.session_state.venue_enc = int(venue_encoder.transform([venue_name])[0])


if 'simulation_started' in st.session_state:
    st.header(f"Simulating Chase: {st.session_state.batting_team_name}")
    
    # Allow user to jump to a specific point in the match
    st.subheader("Set Match State")
    cols = st.columns(3)
    runs_left = cols[0].number_input("Runs Left:", min_value=1, max_value=st.session_state.target, value=st.session_state.runs_left)
    wickets_left = cols[1].number_input("Wickets Left:", min_value=0, max_value=10, value=st.session_state.wickets_left)
    balls_so_far = cols[2].number_input("Balls Bowled:", min_value=0, max_value=119, value=st.session_state.balls_so_far)
    
    # Update session state if user changes these values
    st.session_state.runs_left = runs_left
    st.session_state.wickets_left = wickets_left
    st.session_state.balls_so_far = balls_so_far
    
    # Calculate derived features
    runs_so_far = st.session_state.target - st.session_state.runs_left
    balls_left = 120 - st.session_state.balls_so_far
    current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
    required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 100

    # This dictionary defines the exact feature set for the model using NUMBERS
    features = {
        'batting_team_enc': st.session_state.batting_team_enc,
        'bowling_team_enc': st.session_state.bowling_team_enc,
        'venue_enc': st.session_state.venue_enc,
        'runs_left': st.session_state.runs_left,
        'balls_left': balls_left,
        'wickets_left': st.session_state.wickets_left,
        'target_runs': st.session_state.target,
        'current_run_rate': current_rr,
        'required_run_rate': required_rr,
    }
    
    features_df = pd.DataFrame([features])
    
    # Ensure DataFrame columns are in the correct order before prediction
    training_columns = [
        'batting_team_enc', 'bowling_team_enc', 'venue_enc',
        'runs_left', 'balls_left', 'wickets_left', 'target_runs',
        'current_run_rate', 'required_run_rate'
    ]
    features_df = features_df[training_columns]
    
    current_prob = predict_probability(features_df)
    
    # Display the result using the saved STRING name
    st.header(f"Win Probability: {current_prob * 100:.2f}%")

else:
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")
