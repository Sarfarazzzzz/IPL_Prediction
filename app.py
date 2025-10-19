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

    # Data Cleaning (Consolidated and improved)
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Rising Pune Supergiants': 'Rising Pune Supergiants',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Gujarat Lions': 'Gujarat Lions' # Keep if present in data
    }
    # Comprehensive Venue Mapping (Map variations AND city names if used as venues)
    venue_mapping = {
        # Chinnaswamy
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium', 'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium', 'Bengaluru': 'M Chinnaswamy Stadium',
        # Mohali
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Chandigarh': 'Punjab Cricket Association IS Bindra Stadium', 'Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        # Delhi
        'Feroz Shah Kotla': 'Arun Jaitley Stadium', 'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium', 'Delhi': 'Arun Jaitley Stadium',
        # Mumbai Wankhede
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium', 'Mumbai': 'Wankhede Stadium',
        # Mumbai Brabourne
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium', 'Brabourne Stadium': 'Brabourne Stadium',
         # Mumbai DY Patil
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy', 'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy', 'Navi Mumbai': 'Dr DY Patil Sports Academy',
        # Kolkata
        'Eden Gardens, Kolkata': 'Eden Gardens', 'Kolkata': 'Eden Gardens',
        # Jaipur
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium', 'Jaipur': 'Sawai Mansingh Stadium',
        # Chennai
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium', 'Chennai': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium': 'MA Chidambaram Stadium',
        # Ahmedabad
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium', 'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium', 'Ahmedabad': 'Narendra Modi Stadium',
        # Hyderabad
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium', 'Hyderabad': 'Rajiv Gandhi International Stadium', 'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium',
        # Abu Dhabi
        'Zayed Cricket Stadium, Abu Dhabi': 'Zayed Cricket Stadium', 'Abu Dhabi': 'Zayed Cricket Stadium', 'Sheikh Zayed Stadium': 'Zayed Cricket Stadium',
        # Dharamsala
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium', 'Dharamsala': 'Himachal Pradesh Cricket Association Stadium', 'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium',
         # Pune
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium', 'Pune': 'Maharashtra Cricket Association Stadium', 'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium', 'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
        # Visakhapatnam
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
         # UAE Others
         'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium', 'Dubai': 'Dubai International Cricket Stadium',
         'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium', 'Sharjah': 'Sharjah Cricket Stadium',
         # Others (Add as needed)
         'Barabati Stadium': 'Barabati Stadium', 'Cuttack': 'Barabati Stadium',
         'Holkar Cricket Stadium': 'Holkar Cricket Stadium', 'Indore': 'Holkar Cricket Stadium',
         'JSCA International Stadium Complex': 'JSCA International Stadium Complex', 'Ranchi': 'JSCA International Stadium Complex',
         'Saurashtra Cricket Association Stadium': 'Saurashtra Cricket Association Stadium', 'Rajkot': 'Saurashtra Cricket Association Stadium',
         'Vidarbha Cricket Association Stadium, Jamtha': 'Vidarbha Cricket Association Stadium', 'Nagpur': 'Vidarbha Cricket Association Stadium', 'Vidarbha Cricket Association Stadium': 'Vidarbha Cricket Association Stadium',
         'Green Park': 'Green Park', 'Kanpur': 'Green Park'
    }
    city_mapping = {'Bangalore': 'Bengaluru'} # Keep specific city name changes

    # Apply mappings robustly
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in matches.columns:
            matches[col] = matches[col].astype(str).replace(team_name_mapping) # Ensure string type
    if 'venue' in matches.columns:
        matches['venue'] = matches['venue'].astype(str).replace(venue_mapping) # Ensure string type
    if 'city' in matches.columns:
        matches['city'] = matches['city'].astype(str).replace(city_mapping) # Ensure string type


    return matches

# --- Load Assets ---
model = load_model()
matches_df = load_data()

if model is None or matches_df is None:
    st.error("Failed to load model or data. Cannot proceed.")
    st.stop()

# --- Pre-computation for UI ---
# Add error handling for missing columns
try:
    all_teams = sorted(matches_df['team1'].dropna().unique())
    team_encoding = {team: i for i, team in enumerate(all_teams)}
except KeyError:
    st.error("Column 'team1' not found in matches.csv.")
    all_teams = []
    team_encoding = {}

try:
    all_venues = sorted(matches_df['venue'].dropna().unique())
    venue_encoding = {venue: i for i, venue in enumerate(all_venues)}
except KeyError:
     st.error("Column 'venue' not found in matches.csv.")
     all_venues = []
     venue_encoding = {}

# Correctly handle potential missing cities in home_team_map
home_team_map = {}
if 'city' in matches_df.columns and 'team1' in matches_df.columns:
    try:
        home_team_map = matches_df.dropna(subset=['city', 'team1']).groupby('city')['team1'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).dropna().to_dict()
    except Exception as e:
        st.warning(f"Could not reliably determine home teams: {e}")


# --- Prediction Function: 2-Over Blend, Core Features ONLY ---
def predict_probability(state_df):
    """Predicts win probability using a smooth 2-over blend-in with core features."""
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
    if required_rr > 40 and balls_so_far > 1: # RRR check after first ball
        return 0.0
    # --- End of safety-net rules ---

    # --- 1. Get the ML Model's Prediction (using only core features) ---
    predict_df = state_df.copy()
    predict_df['required_run_rate'] = required_rr # Ensure RRR is in the df

    over = balls_so_far / 6
    predict_df['phase_Middle'] = 1 if 6 < over <= 15 else 0
    predict_df['phase_Death'] = 1 if over > 15 else 0

    # --- THE CORE FIX: Using only reliable, core features ---
    feature_order = [
        'batting_team', 'bowling_team', 'venue', 'balls_so_far', 'balls_left',
        'total_runs_so_far', 'runs_left', 'current_run_rate', 'required_run_rate',
        'wickets_left', 'run_rate_diff', 'is_home_team', 'phase_Middle',
        'phase_Death'
        # 'wicket_pressure' and 'danger_index' are REMOVED
    ]

    # Dynamically get feature names the loaded model expects
    try:
        model_features = model.get_booster().feature_names
    except AttributeError:
        if hasattr(model, 'feature_names_in_'):
             model_features = model.feature_names_in_
        else:
             st.error("Cannot determine expected model features.")
             model_features = feature_order # Best guess

    final_predict_df = pd.DataFrame(columns=model_features) # Empty DF with correct columns

    # Populate DF only with features the model knows and are available
    missing_cols_in_state = []
    for col in model_features:
        if col in predict_df.columns:
            final_predict_df[col] = predict_df[col]
        else:
            # Handle potentially missing columns gracefully (fill with 0)
            missing_cols_in_state.append(col)
            final_predict_df[col] = 0

    # Optional Warning:
    # if missing_cols_in_state:
    #     st.warning(f"Model expected features not found: {missing_cols_in_state}. Filled with 0.")


    final_predict_df.replace([np.inf, -np.inf], 999, inplace=True)
    final_predict_df = final_predict_df.fillna(0) # Ensure no NaNs are passed

    # Make prediction
    try:
        # Ensure DataFrame has at least one row before prediction
        if not final_predict_df.empty:
            model_prob = model.predict_proba(final_predict_df)[0][1]
        else:
            st.warning("Prediction DataFrame is empty. Using fallback probability.")
            model_prob = 0.5 # Fallback
    except Exception as e:
         st.error(f"Error during model prediction: {e}")
         st.write("DataFrame sent to model (first 5 rows):", final_predict_df.head())
         model_prob = 0.5 # Fallback


    # Apply Confidence Multiplier for low-wicket scenarios
    if wickets_left <= 3:
        confidence_multipliers = {1: 0.5, 2: 0.75, 3: 0.9}
        multiplier = confidence_multipliers.get(wickets_left, 1.0)
        model_prob *= multiplier

    # --- 2. Transition (Overs 1-2 / Balls 1-12): Blend Heuristic & Model ---
    if balls_so_far <= 12:
        initial_prob = st.session_state.get('initial_prob', 0.5) # Get initial prob safely
        model_weight = max(0, balls_so_far) / 12.0 # Blend weight from 0 to 1

        final_prob = ((1 - model_weight) * initial_prob) + (model_weight * model_prob)
        # Clip final probability during blend
        return np.clip(final_prob, 0.01, 0.99)


    # --- 3. Full Model Control (Over 3+ / Ball 13+): Use 100% ML Model ---
    else:
         # Clip final probability after blend phase
        return np.clip(model_prob, 0.01, 0.99)


# --- UI Layout ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("A stable start that smoothly transitions to a live in-game model.")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Match Setup")
    if not all_teams:
        st.error("Cannot proceed without team data.")
        st.stop()
    batting_team = st.selectbox("Select Batting Team", all_teams, key="sb_bat_team")

    available_bowling_teams = [t for t in all_teams if t != batting_team]
    if not available_bowling_teams:
         # If only one team, allow selecting itself? Or handle differently?
         # For now, allow selecting the same team, although logically incorrect for a match.
         st.warning("Only one team available. Please check data.")
         available_bowling_teams = all_teams

    bowling_team = st.selectbox("Select Bowling Team", available_bowling_teams, key="sb_bowl_team", index=min(1, len(available_bowling_teams)-1) if len(available_bowling_teams)>1 else 0)


    # --- City / Venue Selection ---
    selected_city = None
    venue = None
    possible_venues = []

    valid_cities = sorted(matches_df['city'].dropna().unique()) if 'city' in matches_df.columns else []

    if valid_cities:
        selected_city = st.selectbox("Select City", valid_cities, key="sb_city")
        if 'venue' in matches_df.columns:
            possible_venues = sorted(matches_df[matches_df['city'] == selected_city]['venue'].dropna().unique())
            if not possible_venues: # Fallback if no venues for city
                 possible_venues = sorted(matches_df['venue'].dropna().unique())
                 st.warning(f"No specific venues found for {selected_city}, showing all venues.")
        else:
            st.error("Column 'venue' not found.")
            st.stop() # Cannot proceed without venue data
    else:
        st.warning("No city data found. Select venue directly.")
        if 'venue' in matches_df.columns:
            possible_venues = sorted(matches_df['venue'].dropna().unique())
        else:
             st.error("Column 'venue' not found.")
             st.stop() # Cannot proceed without venue data

    if possible_venues:
         venue = st.selectbox("Select Venue", possible_venues, key="sb_venue")
    else:
        st.error("No venues available for selection.")
        st.stop() # Cannot proceed without venues


    target_runs = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180, key="sb_target")

    # --- Start Button Logic ---
    if st.button("Start / Reset Simulation", type="primary", key="btn_start"):
        if not batting_team or not bowling_team or not venue:
             st.warning("Please ensure batting team, bowling team, and venue are selected.")
        elif batting_team == bowling_team:
             st.warning("Batting and Bowling teams cannot be the same.")
        else:
            # Clear previous state
            for key in list(st.session_state.keys()):
                if key not in ['sb_bat_team', 'sb_bowl_team', 'sb_city', 'sb_venue', 'sb_target']: # Keep sidebar selections
                     del st.session_state[key]

            # Initialize new state
            st.session_state.simulation_started = True
            st.session_state.target = target_runs
            st.session_state.runs_left = target_runs
            st.session_state.wickets_left = 10
            st.session_state.balls_so_far = 0
            st.session_state.batting_team = batting_team
            st.session_state.bowling_team = bowling_team
            st.session_state.venue = venue
            st.session_state.selected_city = selected_city

            # Calculate target-sensitive initial probability
            par_score = 175
            score_diff = par_score - target_runs
            adjustment = score_diff * 0.0075
            initial_prob = 0.5 + adjustment
            initial_prob = np.clip(initial_prob, 0.1, 0.9) # Keep within 10%-90%

            st.session_state.initial_prob = initial_prob
            st.session_state.probabilities = [initial_prob] # Start list with initial probability
            st.session_state.overs_history = [0.0]        # Start list at over 0.0
            st.rerun()

# --- Main Page Display ---
if st.session_state.get('simulation_started', False):
    st.header("Current Match State")
    col1, col2, col3, col4 = st.columns(4)
    # Use .get with default values
    col1.metric("Target", st.session_state.get('target', 'N/A'))
    col2.metric("Runs Left", st.session_state.get('runs_left', 'N/A'))
    col3.metric("Wickets Left", st.session_state.get('wickets_left', 'N/A'))
    balls_so_far_main = st.session_state.get('balls_so_far', 0)
    col4.metric("Balls Left", 120 - balls_so_far_main)

    # --- Jump State Expander ---
    with st.expander("✏️ Jump to a Specific Point in the Match"):
        override_cols = st.columns(3)
        # Defaults for jump inputs
        default_over = int(st.session_state.get('balls_so_far', 0) / 6)
        default_ball = st.session_state.get('balls_so_far', 0) % 6
        default_runs_left = st.session_state.get('runs_left', st.session_state.get('target', 180))
        default_wickets_left = st.session_state.get('wickets_left', 10)
        max_target_runs = st.session_state.get('target', 400)

        over_input = override_cols[0].number_input("Overs Bowled:", min_value=0, max_value=20, value=default_over, step=1, key="jump_over_exp")
        ball_input = override_cols[0].number_input("Balls in Over:", min_value=0, max_value=5, value=default_ball, step=1, key="jump_ball_exp")
        runs_left_input = override_cols[1].number_input("Set Runs Left:", min_value=0, max_value=max_target_runs, value=default_runs_left, key="jump_runs_exp")
        wickets_left_input = override_cols[2].number_input("Set Wickets Left:", min_value=0, max_value=10, value=default_wickets_left, key="jump_wickets_exp")

        if st.button("Apply Custom State", key="btn_jump"):
            new_balls_so_far = (over_input * 6) + ball_input
            # Validate inputs
            if new_balls_so_far < 0 or new_balls_so_far > 120 or \
               runs_left_input < 0 or runs_left_input > max_target_runs or \
               wickets_left_input < 0 or wickets_left_input > 10:
                 st.error("Invalid input values for jumping to state.")
            else:
                st.session_state.balls_so_far = new_balls_so_far
                st.session_state.runs_left = runs_left_input
                st.session_state.wickets_left = wickets_left_input

                balls_left = 120 - st.session_state.balls_so_far
                target = st.session_state.get('target', runs_left_input)
                runs_so_far = max(0, target - st.session_state.runs_left)
                current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
                required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0

                current_batting_team = st.session_state.get('batting_team', all_teams[0])
                current_bowling_team = st.session_state.get('bowling_team', all_teams[1])
                current_venue = st.session_state.get('venue', all_venues[0])
                current_city = st.session_state.get('selected_city', "Unknown")

                is_home = 0
                if home_team_map and current_city != "Unknown" and current_batting_team == home_team_map.get(current_city):
                     is_home = 1

                state_df = pd.DataFrame([{
                    'batting_team': team_encoding.get(current_batting_team),
                    'bowling_team': team_encoding.get(current_bowling_team),
                    'venue': venue_encoding.get(current_venue),
                    'balls_so_far': st.session_state.balls_so_far,
                    'balls_left': balls_left,
                    'total_runs_so_far': runs_so_far,
                    'runs_left': st.session_state.runs_left,
                    'current_run_rate': current_rr,
                    'required_run_rate': required_rr,
                    'wickets_left': st.session_state.wickets_left,
                    'run_rate_diff': current_rr - required_rr,
                    'is_home_team': is_home
                }])

                prob_at_jump = predict_probability(state_df)
                st.session_state.probabilities = [prob_at_jump] # Reset history
                st.session_state.overs_history = [st.session_state.balls_so_far / 6]
                st.rerun()

    # --- Ball Input Section ---
    st.divider()
    st.header("Ball-by-Ball Input")
    input_col, plot_col = st.columns([1, 2])

    with input_col:
        runs_scored = st.selectbox("Runs on this ball:", (0, 1, 2, 3, 4, 6), key="runs_input")
        is_wicket = st.checkbox("Wicket on this ball?", key="wicket_input")

        if st.button("Next Ball", type="secondary", key="btn_next_ball"):
            if 'wickets_left' not in st.session_state or 'runs_left' not in st.session_state or 'balls_so_far' not in st.session_state:
                 st.warning("Please start or reset the simulation first.")
            elif st.session_state.wickets_left <= 0 or st.session_state.runs_left <= 0 or st.session_state.balls_so_far >= 120:
                st.warning("Match is over! Please reset the simulation to start a new one.")
            else:
                # --- Update State ---
                runs_this_ball = runs_scored if isinstance(runs_scored, int) else 0 # Ensure runs is int
                st.session_state.runs_left = max(0, st.session_state.runs_left - runs_this_ball)
                st.session_state.balls_so_far += 1
                if is_wicket:
                    st.session_state.wickets_left -= 1

                # --- Recalculate derived metrics ---
                balls_left = 120 - st.session_state.balls_so_far
                target = st.session_state.get('target', st.session_state.runs_left)
                runs_so_far = max(0, target - st.session_state.runs_left)
                current_rr = (runs_so_far * 6 / st.session_state.balls_so_far) if st.session_state.balls_so_far > 0 else 0
                required_rr = (st.session_state.runs_left * 6 / balls_left) if balls_left > 0 else 0

                # --- Get current context safely ---
                current_batting_team = st.session_state.get('batting_team', all_teams[0])
                current_bowling_team = st.session_state.get('bowling_team', all_teams[1])
                current_venue = st.session_state.get('venue', all_venues[0])
                current_city = st.session_state.get('selected_city', "Unknown")

                is_home = 0
                if home_team_map and current_city != "Unknown" and current_batting_team == home_team_map.get(current_city):
                    is_home = 1


                # --- Create DataFrame for prediction ---
                state_df = pd.DataFrame([{
                    'batting_team': team_encoding.get(current_batting_team),
                    'bowling_team': team_encoding.get(current_bowling_team),
                    'venue': venue_encoding.get(current_venue),
                    'balls_so_far': st.session_state.balls_so_far,
                    'balls_left': balls_left,
                    'total_runs_so_far': runs_so_far,
                    'runs_left': st.session_state.runs_left,
                    'current_run_rate': current_rr,
                    'required_run_rate': required_rr,
                    'wickets_left': st.session_state.wickets_left,
                    'run_rate_diff': current_rr - required_rr,
                    'is_home_team': is_home
                }])

                # --- Predict and update history ---
                win_prob = predict_probability(state_df)

                # Ensure lists exist before appending
                if 'probabilities' not in st.session_state: st.session_state.probabilities = [st.session_state.initial_prob]
                if 'overs_history' not in st.session_state: st.session_state.overs_history = [0.0]

                st.session_state.probabilities.append(win_prob)
                st.session_state.overs_history.append(st.session_state.balls_so_far / 6)
                st.rerun()

    # --- Plotting Section ---
    with plot_col:
        st.subheader("Win Probability Chart")
        if st.session_state.get('probabilities') and st.session_state.get('overs_history'):
            try:
                # Ensure lists are not empty and lengths match
                probs = st.session_state.probabilities
                overs = st.session_state.overs_history
                min_len = min(len(overs), len(probs))

                if min_len < 1: # Need at least one point to plot
                    st.info("Not enough data points yet.")
                else:
                    overs_hist = overs[:min_len]
                    bat_probs = np.array(probs[:min_len])
                    bowl_probs = 1 - bat_probs

                    fig, ax = plt.subplots(figsize=(10, 6))

                    batting_label = st.session_state.get('batting_team', 'Batting Team')
                    bowling_label = st.session_state.get('bowling_team', 'Bowling Team')

                    ax.plot(overs_hist, bat_probs, label=batting_label, color="#0072B2", linewidth=2.5, marker='o', markersize=5)
                    ax.plot(overs_hist, bowl_probs, label=bowling_label, color="#D55E00", linewidth=2.5, marker='o', markersize=5)

                    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.8)
                    ax.set_xlabel("Overs"); ax.set_ylabel("Win Probability"); ax.set_title("Live Win Probability", fontsize=16)
                    ax.set_ylim(0, 1); ax.set_xlim(min(overs_hist or [0]) - 1, 20); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    ax.legend()
                    st.pyplot(fig)

                    final_prob_val = bat_probs[-1] * 100
                    st.metric(label=f"**{batting_label}'s Current Win Probability**", value=f"{final_prob_val:.2f}%")

            except Exception as e:
                st.error(f"Error plotting chart: {e}")
                st.write("Probabilities:", st.session_state.get('probabilities', []))
                st.write("Overs History:", st.session_state.get('overs_history', []))
        else:
             st.info("Waiting for simulation data.")

# --- Initial State Message ---
else:
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")
