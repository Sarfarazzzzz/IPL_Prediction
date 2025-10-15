import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- Caching Functions (No Changes) ---
# These functions load the model and data efficiently.
@st.cache_resource
def load_model():
    try:
        with open("lgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'lgb_model.pkl' not found.")
        return None

@st.cache_data
def load_data():
    try:
        matches_df = pd.read_csv("matches.csv")
    except FileNotFoundError:
        st.error("Matches file 'matches.csv' not found.")
        return None
    
    # Data Cleaning
    team_name_mapping = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings', 'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants', 'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'}
    team_cols = ['team1', 'team2', 'toss_winner', 'winner']
    for col in team_cols:
        matches_df[col] = matches_df[col].replace(team_name_mapping)
    
    venue_mapping = {'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium', 'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium', 'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Feroz Shah Kotla': 'Arun Jaitley Stadium', 'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium', 'Wankhede Stadium, Mumbai': 'Wankhede Stadium', 'Brabourne Stadium, Mumbai': 'Brabourne Stadium', 'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy', 'Eden Gardens, Kolkata': 'Eden Gardens', 'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium', 'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Uppal', 'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium', 'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium', 'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium'}
    matches_df['venue'] = matches_df['venue'].replace(venue_mapping)
    
    city_mapping = {'Bangalore': 'Bengaluru'}
    matches_df['city'] = matches_df['city'].replace(city_mapping)
    
    return matches_df

# --- Load Assets ---
model = load_model()
matches_df = load_data()

if model is None or matches_df is None:
    st.stop()

# --- Pre-computation (No Changes) ---
all_teams = sorted(matches_df['team1'].dropna().unique())
team_encoding = {team: i for i, team in enumerate(all_teams)}
all_venues = sorted(matches_df['venue'].dropna().unique())
venue_encoding = {venue: i for i, venue in enumerate(all_venues)}
city_to_home_team = {row['city']: row['team1'] for index, row in matches_df.dropna(subset=['city', 'team1']).iterrows()}

# --- NEW: Prediction Function ---
# This function calculates win probability for a given match state.
def predict_probability(match_state):
    features = pd.DataFrame([match_state], columns=[
        'batting_team', 'bowling_team', 'venue', 'balls_so_far', 'balls_left',
        'total_runs_so_far', 'runs_left', 'current_run_rate', 'required_run_rate',
        'wickets_left', 'run_rate_diff', 'is_home_team'
    ])
    return model.predict_proba(features)[0][1]

# --- Streamlit UI ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("Simulate a match ball-by-ball and see the win probability change in real-time.")

# --- Sidebar for Match Setup ---
with st.sidebar:
    st.header("⚙️ Match Setup")
    batting_team = st.selectbox("Select Batting Team", all_teams)
    bowling_team = st.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team])
    sorted_cities = sorted(matches_df['city'].dropna().unique())
    selected_city = st.selectbox("Select City", sorted_cities)
    possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].unique())
    venue = st.selectbox("Select Venue", possible_venues)
    target_runs = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180)

    # --- NEW: Start and Reset Buttons ---
    if st.button("Start / Reset Simulation", type="primary"):
        # Initialize the session state for a new simulation
        st.session_state.simulation_started = True
        st.session_state.target = target_runs
        st.session_state.runs_left = target_runs
        st.session_state.wickets_left = 10
        st.session_state.balls_so_far = 0
        st.session_state.batting_team = batting_team
        st.session_state.bowling_team = bowling_team
        st.session_state.probabilities = []
        st.session_state.overs_history = []
        
        # Store encoded values in state
        st.session_state.batting_team_enc = team_encoding[batting_team]
        st.session_state.bowling_team_enc = team_encoding[bowling_team]
        st.session_state.venue_enc = venue_encoding[venue]
        st.session_state.is_home_team = 1 if batting_team == city_to_home_team.get(selected_city) else 0

# --- NEW: Main Interactive Area ---
# This section only appears after the simulation has started.
if 'simulation_started' in st.session_state and st.session_state.simulation_started:
    
    # Display current match state using metrics
    st.header("Current Match State")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", st.session_state.target)
    col2.metric("Runs Left", st.session_state.runs_left)
    col3.metric("Wickets Left", st.session_state.wickets_left)
    col4.metric("Balls Left", 120 - st.session_state.balls_so_far)

    st.divider()

    # Create columns for user input
    st.header("Ball-by-Ball Input")
    input_col, plot_col = st.columns([1, 2])

    with input_col:
        runs_scored = st.selectbox("Runs on this ball:", (0, 1, 2, 3, 4, 6), key="runs")
        is_wicket = st.checkbox("Wicket on this ball?", key="wicket")
        
        if st.button("Next Ball", type="secondary"):
            if st.session_state.wickets_left == 0 or st.session_state.runs_left <= 0 or st.session_state.balls_so_far >= 120:
                st.warning("Match is over! Please reset the simulation to start a new one.")
            else:
                # Update state based on input
                st.session_state.runs_left -= runs_scored
                st.session_state.balls_so_far += 1
                if is_wicket:
                    st.session_state.wickets_left -= 1
                
                # Calculate derived features
                balls_left = 120 - st.session_state.balls_so_far
                runs_so_far = st.session_state.target - st.session_state.runs_left
                current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
                required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 100
                
                # Create the match state dictionary for prediction
                current_state = {
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
                    'is_home_team': st.session_state.is_home_team
                }
                
                # Get new probability and add to history
                win_prob = predict_probability(current_state)
                st.session_state.probabilities.append(win_prob)
                st.session_state.overs_history.append(st.session_state.balls_so_far / 6)

    with plot_col:
        st.subheader("Win Probability Chart")
        if st.session_state.probabilities:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batting_team_probs = np.array(st.session_state.probabilities)
            bowling_team_probs = 1 - batting_team_probs
            
            ax.plot(st.session_state.overs_history, batting_team_probs, label=f"{st.session_state.batting_team}", color="#0072B2", linewidth=2.5)
            ax.plot(st.session_state.overs_history, bowling_team_probs, label=f"{st.session_state.bowling_team}", color="#D55E00", linewidth=2.5)

            ax.axhline(0.5, linestyle="--", color="grey", alpha=0.8)
            
            ax.set_xlabel("Overs")
            ax.set_ylabel("Win Probability")
            ax.set_title("Live Win Probability", fontsize=16)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 20)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display the final prediction
            final_prob = batting_team_probs[-1] * 100
            st.metric(label=f"**{st.session_state.batting_team}'s Current Win Probability**", value=f"{final_prob:.2f}%")

else:
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")










