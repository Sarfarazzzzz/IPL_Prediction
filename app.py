import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Load All Model Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open("lgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("team_encoder.pkl", "rb") as f:
            team_encoder = pickle.load(f)
        with open("venue_encoder.pkl", "rb") as f:
            venue_encoder = pickle.load(f)
        return model, team_encoder, venue_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure lgb_model.pkl, team_encoder.pkl, and venue_encoder.pkl are in the GitHub repository.")
        return None, None, None

model, team_encoder, venue_encoder = load_artifacts()

@st.cache_data
def load_match_data():
    try:
        matches_df = pd.read_csv("matches.csv")
        # Perform the same cleaning as in the notebook
        team_name_mapping = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings', 'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants', 'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'}
        matches_df['team1'] = matches_df['team1'].replace(team_name_mapping)
        matches_df['team2'] = matches_df['team2'].replace(team_name_mapping)
        city_mapping = {'Bangalore': 'Bengaluru'}
        matches_df['city'] = matches_df['city'].replace(city_mapping)
        return matches_df
    except FileNotFoundError:
        st.error("Matches file 'matches.csv' not found.")
        return None

matches_df = load_match_data()

if not all([model, team_encoder, venue_encoder, matches_df is not None]):
    st.stop()

# --- Get UI Lists from Encoders/Data ---
all_teams = sorted(team_encoder.classes_)
all_venues = sorted(venue_encoder.classes_)
all_cities = sorted(matches_df['city'].dropna().unique())
city_to_home_team = {row['city']: row['team1'] for index, row in matches_df.dropna(subset=['city', 'team1']).iterrows()}

# --- Prediction Function ---
def predict_probability(match_state):
    df = pd.DataFrame([match_state])
    over = df['balls_so_far'].iloc[0] / 6
    df['phase_Middle'] = 1 if 6 < over <= 15 else 0
    df['phase_Death'] = 1 if over > 15 else 0
    df['wicket_pressure'] = df['required_run_rate'] * (11 - df['wickets_left'])
    
    feature_order = [
        'batting_team', 'bowling_team', 'venue', 'balls_so_far', 'balls_left',
        'total_runs_so_far', 'runs_left', 'current_run_rate', 'required_run_rate',
        'wickets_left', 'run_rate_diff', 'is_home_team', 'phase_Middle', 'phase_Death', 'wicket_pressure'
    ]
    df_final = df[feature_order]
    return model.predict_proba(df_final.values)[0][1]

# --- UI Setup ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("Simulate a match ball-by-ball or jump to any point in the chase.")

with st.sidebar:
    st.header("⚙️ Match Setup")
    batting_team_name = st.selectbox("Select Batting Team", all_teams)
    bowling_team_name = st.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team_name])
    selected_city = st.selectbox("Select City", all_cities)
    possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].unique())
    venue_name = st.selectbox("Select Venue", possible_venues if len(possible_venues) > 0 else all_venues)
    target_runs = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180)

    if st.button("Start / Reset Simulation", type="primary"):
        st.session_state.clear()
        st.session_state.simulation_started = True
        st.session_state.target = target_runs
        st.session_state.runs_left = target_runs
        st.session_state.wickets_left = 10
        st.session_state.balls_so_far = 0
        st.session_state.batting_team = batting_team_name
        st.session_state.bowling_team = bowling_team_name
        st.session_state.probabilities = []
        st.session_state.overs_history = []
        
        st.session_state.batting_team_enc = team_encoder.transform([batting_team_name])[0]
        st.session_state.bowling_team_enc = team_encoder.transform([bowling_team_name])[0]
        st.session_state.venue_enc = venue_encoder.transform([venue_name])[0]
        st.session_state.is_home_team = 1 if batting_team_name == city_to_home_team.get(selected_city) else 0

# --- Main App Body ---
if 'simulation_started' in st.session_state:
    st.header("Current Match State")
    # ... (Rest of the UI and logic is the same, no changes needed below this line)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", st.session_state.target)
    col2.metric("Runs Left", st.session_state.runs_left)
    col3.metric("Wickets Left", st.session_state.wickets_left)
    col4.metric("Balls Left", 120 - st.session_state.balls_so_far)

    with st.expander("✏️ Jump to a Specific Point in the Match"):
        override_cols = st.columns(3)
        over_input = override_cols[0].number_input("Overs Bowled:", min_value=0, max_value=20, value=int(st.session_state.balls_so_far / 6), step=1)
        ball_input = override_cols[0].number_input("Balls in Over:", min_value=0, max_value=5, value=st.session_state.balls_so_far % 6, step=1)
        runs_left_input = override_cols[1].number_input("Set Runs Left:", min_value=1, max_value=st.session_state.target, value=st.session_state.runs_left)
        wickets_left_input = override_cols[2].number_input("Set Wickets Left:", min_value=0, max_value=10, value=st.session_state.wickets_left)

        if st.button("Apply Custom State"):
            st.session_state.balls_so_far = (over_.input * 6) + ball_input
            st.session_state.runs_left = runs_left_input
            st.session_state.wickets_left = wickets_left_input
            
            balls_left = 120 - st.session_state.balls_so_far
            runs_so_far = st.session_state.target - st.session_state.runs_left
            current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
            required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0
            
            initial_state = {
                'batting_team': st.session_state.batting_team_enc, 'bowling_team': st.session_state.bowling_team_enc,
                'venue': st.session_state.venue_enc, 'balls_so_far': st.session_state.balls_so_far,
                'balls_left': balls_left, 'total_runs_so_far': runs_so_far,
                'runs_left': st.session_state.runs_left, 'current_run_rate': current_rr,
                'required_run_rate': required_rr, 'wickets_left': st.session_state.wickets_left,
                'run_rate_diff': current_rr - required_rr, 'is_home_team': st.session_state.is_home_team
            }
            
            initial_prob = predict_probability(initial_state)
            st.session_state.probabilities = [initial_prob]
            st.session_state.overs_history = [st.session_state.balls_so_far / 6]
            st.success("Match state updated! Initial probability calculated.")
            st.rerun()

    st.divider()
    st.header("Ball-by-Ball Input")
    input_col, plot_col = st.columns([1, 2])

    with input_col:
        runs_scored = st.selectbox("Runs on this ball:", (0, 1, 2, 3, 4, 6), key="runs")
        is_wicket = st.checkbox("Wicket on this ball?", key="wicket")
        
        if st.button("Next Ball", type="secondary"):
            if st.session_state.wickets_left <= 0 or st.session_state.runs_left <= 0 or st.session_state.balls_so_far >= 120:
                st.warning("Match is over! Please reset the simulation to start a new one.")
            else:
                st.session_state.runs_left -= runs_scored
                st.session_state.balls_so_far += 1
                if is_wicket:
                    st.session_state.wickets_left -= 1
                
                balls_left = 120 - st.session_state.balls_so_far
                runs_so_far = st.session_state.target - st.session_state.runs_left
                current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
                required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0
                
                current_state = {
                    'batting_team': st.session_state.batting_team_enc, 'bowling_team': st.session_state.bowling_team_enc,
                    'venue': st.session_state.venue_enc, 'balls_so_far': st.session_state.balls_so_far,
                    'balls_left': balls_left, 'total_runs_so_far': runs_so_far,
                    'runs_left': st.session_state.runs_left, 'current_run_rate': current_rr,
                    'required_run_rate': required_rr, 'wickets_left': st.session_state.wickets_left,
                    'run_rate_diff': current_rr - required_rr, 'is_home_team': st.session_state.is_home_team
                }
                
                win_prob = predict_probability(current_state)
                st.session_state.probabilities.append(win_prob)
                st.session_state.overs_history.append(st.session_state.balls_so_far / 6)
                st.rerun()

    with plot_col:
        st.subheader("Win Probability Chart")
        if st.session_state.probabilities:
            fig, ax = plt.subplots(figsize=(10, 6))
            batting_team_probs = np.array(st.session_state.probabilities)
            bowling_team_probs = 1 - batting_team_probs
            ax.plot(st.session_state.overs_history, batting_team_probs, label=f"{st.session_state.batting_team}", color="#0072B2", linewidth=2.5, marker='o', markersize=5)
            ax.plot(st.session_state.overs_history, bowling_team_probs, label=f"{st.session_state.bowling_team}", color="#D55E00", linewidth=2.5, marker='o', markersize=5)
            ax.axhline(0.5, linestyle="--", color="grey", alpha=0.8)
            ax.set_xlabel("Overs"); ax.set_ylabel("Win Probability"); ax.set_title("Live Win Probability", fontsize=16)
            ax.set_ylim(0, 1); ax.set_xlim(min(st.session_state.overs_history or [0]) - 1, 20); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            st.pyplot(fig)
            
            final_prob = batting_team_probs[-1] * 100
            st.metric(label=f"**{st.session_state.batting_team}'s Current Win Probability**", value=f"{final_prob:.2f}%")
else:
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")



