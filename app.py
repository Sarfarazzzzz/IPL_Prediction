import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- Caching Functions ---
@st.cache_resource
def load_model():
    """Loads the final, context-aware prediction model."""
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'xgb_model.pkl' not found.")
        return None

@st.cache_data
def load_data():
    """Loads and cleans the base matches data."""
    try:
        matches = pd.read_csv("matches.csv")
    except FileNotFoundError:
        st.error("Required data file 'matches.csv' not found.")
        return None

    # Data Cleaning
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
    }
    venue_mapping = {
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium', 'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium', 'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium', 'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy', 'Eden Gardens, Kolkata': 'Eden Gardens',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium', 'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium', 'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
        'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Zayed Cricket Stadium, Abu Dhabi': 'Zayed Cricket Stadium', 'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'
    }
    city_mapping = {'Bangalore': 'Bengaluru'}

    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        matches[col] = matches[col].replace(team_name_mapping)
    matches['venue'] = matches['venue'].replace(venue_mapping)
    matches['city'] = matches['city'].replace(city_mapping)

    return matches

# --- Load Assets ---
model = load_model()
matches_df = load_data()

if model is None or matches_df is None:
    st.stop()

# --- Pre-computation for UI ---
all_teams = sorted(matches_df['team1'].dropna().unique())
team_encoding = {team: i for i, team in enumerate(all_teams)}
all_venues = sorted(matches_df['venue'].dropna().unique())
venue_encoding = {venue: i for i, venue in enumerate(all_venues)}
home_team_map = matches_df.groupby('city')['team1'].agg(lambda x: x.value_counts().index[0]).to_dict()

# --- Prediction Function with 1-Over Lock + 1-Over Blend ---
def predict_probability(state_df):
    """Predicts win probability using a 1-over lock & 1-over blend."""
    balls_so_far = state_df['balls_so_far'].iloc[0]
    wickets_left = state_df['wickets_left'].iloc[0]
    runs_left = state_df['runs_left'].iloc[0]
    balls_left = state_df['balls_left'].iloc[0]
    
    required_rr = (runs_left * 6) / balls_left if balls_left > 0 else 999

    # --- Safety-net rules ---
    if runs_left <= 0:
        return 1.0
    if wickets_left <= 0 and runs_left > 0:
        return 0.0
    if balls_left <= 0 and runs_left > 0:
        return 0.0
    if required_rr > 40 and balls_so_far > 1: 
        return 0.0
    # --- End of safety-net rules ---

    # --- 1. First Over (Balls 1-6): Lock to Initial Probability ---
    if balls_so_far <= 6:
        return st.session_state.initial_prob

    # --- 2. Get the ML Model's Prediction (for Over 2 onwards) ---
    predict_df = state_df.copy()
    predict_df['required_run_rate'] = required_rr

    # Use the ORIGINAL feature formulas, as the model was trained on them
    predict_df['wicket_pressure'] = required_rr * (11 - wickets_left)
    predict_df['danger_index'] = required_rr / (wickets_left + 0.1)
    
    over = balls_so_far / 6
    predict_df['phase_Middle'] = 1 if 6 < over <= 15 else 0
    predict_df['phase_Death'] = 1 if over > 15 else 0
    
    feature_order = [
        'batting_team', 'bowling_team', 'venue', 'balls_so_far', 'balls_left',
        'total_runs_so_far', 'runs_left', 'current_run_rate', 'required_run_rate',
        'wickets_left', 'run_rate_diff', 'is_home_team', 'phase_Middle',
        'phase_Death', 'wicket_pressure', 'danger_index'
    ]
    
    final_predict_df = pd.DataFrame(columns=model.get_booster().feature_names)
    for col in feature_order:
         if col in final_predict_df.columns:
            final_predict_df[col] = predict_df[col]

    final_predict_df.replace([np.inf, -np.inf], 999, inplace=True)
    final_predict_df.fillna(0, inplace=True)
    
    model_prob = model.predict_proba(final_predict_df)[0][1]

    # Apply Confidence Multiplier
    if wickets_left <= 3:
        confidence_multipliers = {1: 0.5, 2: 0.75, 3: 0.9}
        multiplier = confidence_multipliers.get(wickets_left, 1.0)
        model_prob *= multiplier

    # --- 3. Transition (Over 2 / Balls 7-12): Blend Heuristic & Model ---
    if 6 < balls_so_far <= 12:
        # Get the stable starting prob
        initial_prob = st.session_state.initial_prob
        
        # Calculate the blend weight. It increases from 1/6 to 6/6 (1.0)
        model_weight = (balls_so_far - 6) / 6.0
        
        final_prob = ((1 - model_weight) * initial_prob) + (model_weight * model_prob)
        return final_prob

    # --- 4. Full Model Control (Over 3+ / Ball 13+): Use 100% ML Model ---
    else:
        return model_prob

# --- UI Layout ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("A stable, target-sensitive start that smoothly transitions to a live in-game model.")

with st.sidebar:
    st.header("⚙️ Match Setup")
    batting_team = st.selectbox("Select Batting Team", all_teams)
    bowling_team = st.selectbox("Select Bowling Team", [t for t in all_teams if t != batting_team])
    sorted_cities = sorted(matches_df['city'].dropna().unique())
    selected_city = st.selectbox("Select City", sorted_cities)
    possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].dropna().unique())
    venue = st.selectbox("Select Venue", possible_venues)
    target_runs = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180)

    if st.button("Start / Reset Simulation", type="primary"):
        st.session_state.clear()
        st.session_state.simulation_started = True
        st.session_state.target = target_runs
        st.session_state.runs_left = target_runs
        st.session_state.wickets_left = 10
        st.session_state.balls_so_far = 0
        st.session_state.batting_team = batting_team
        st.session_state.bowling_team = bowling_team
        st.session_state.venue = venue
        st.session_state.selected_city = selected_city

        par_score = 175
        score_diff = par_score - target_runs
        adjustment = score_diff * 0.0075 
        initial_prob = 0.5 + adjustment
        initial_prob = np.clip(initial_prob, 0.1, 0.9)
        
        st.session_state.initial_prob = initial_prob
        st.session_state.probabilities = [initial_prob]
        st.session_state.overs_history = [0.0]
        st.rerun()

if 'simulation_started' in st.session_state and st.session_state.simulation_started:
    st.header("Current Match State")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", st.session_state.target)
    col2.metric("Runs Left", st.session_state.runs_left)
    col3.metric("Wickets Left", st.session_state.wickets_left)
    col4.metric("Balls Left", 120 - st.session_state.balls_so_far)

    with st.expander("✏️ Jump to a Specific Point in the Match"):
        override_cols = st.columns(3)
        over_input = override_cols[0].number_input("Overs Bowled:", min_value=0, max_value=20, value=int(st.session_state.balls_so_far / 6), step=1)
        ball_input = override_cols[0].number_input("Balls in Over:", min_value=0, max_value=5, value=st.session_state.balls_so_far % 6, step=1)
        runs_left_input = override_cols[1].number_input("Set Runs Left:", min_value=0, max_value=st.session_state.target, value=st.session_state.runs_left)
        wickets_left_input = override_cols[2].number_input("Set Wickets Left:", min_value=0, max_value=10, value=st.session_state.wickets_left)

        if st.button("Apply Custom State"):
            st.session_state.balls_so_far = (over_input * 6) + ball_input
            st.session_state.runs_left = runs_left_input
            st.session_state.wickets_left = wickets_left_input
            
            balls_left = 120 - st.session_state.balls_so_far
            runs_so_far = st.session_state.target - st.session_state.runs_left
            current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
            required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0
            
            state_df = pd.DataFrame([{
                'batting_team': team_encoding.get(st.session_state.batting_team),
                'bowling_team': team_encoding.get(st.session_state.bowling_team),
                'venue': venue_encoding.get(st.session_state.venue),
                'balls_so_far': st.session_state.balls_so_far,
                'balls_left': balls_left,
                'total_runs_so_far': runs_so_far,
                'runs_left': st.session_state.runs_left,
                'current_run_rate': current_rr,
                'required_run_rate': required_rr,
                'wickets_left': st.session_state.wickets_left,
                'run_rate_diff': current_rr - required_rr,
                'is_home_team': 1 if st.session_state.batting_team == home_team_map.get(st.session_state.selected_city) else 0
            }])
            
            prob_at_jump = predict_probability(state_df)
            st.session_state.probabilities = [prob_at_jump]
            st.session_state.overs_history = [st.session_state.balls_so_far / 6]
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
                st.session_state.runs_left = max(0, st.session_state.runs_left - runs_scored)
                st.session_state.balls_so_far += 1
                if is_wicket:
                    st.session_state.wickets_left -= 1
                
                balls_left = 120 - st.session_state.balls_so_far
                runs_so_far = st.session_state.target - st.session_state.runs_left
                current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
                required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0
                
                state_df = pd.DataFrame([{
                    'batting_team': team_encoding.get(st.session_state.batting_team),
                    'bowling_team': team_encoding.get(st.session_state.bowling_team),
                    'venue': venue_encoding.get(st.session_state.venue),
                    'balls_so_far': st.session_state.balls_so_far,
                    'balls_left': balls_left,
                    'total_runs_so_far': runs_so_far,
                    'runs_left': st.session_state.runs_left,
                    'current_run_rate': current_rr,
                    'required_run_rate': required_rr,
                    'wickets_left': st.session_state.wickets_left,
                    'run_rate_diff': current_rr - required_rr,
                    'is_home_team': 1 if st.session_state.batting_team == home_team_map.get(st.session_state.selected_city) else 0
                }])
                
                win_prob = predict_probability(state_df)
                st.session_state.probabilities.append(win_prob)
                st.session_state.overs_history.append(st.session_state.balls_so_far / 6)
                st.rerun()
                
    with plot_col:
        st.subheader("Win Probability Chart")
        if 'probabilities' in st.session_state and len(st.session_state.probabilities) > 0:
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
