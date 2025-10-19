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
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():
    """Loads and cleans the base matches data."""
    try:
        matches = pd.read_csv("matches.csv")
        required_cols = ['team1', 'team2', 'venue', 'city', 'winner', 'toss_winner']
        if not all(col in matches.columns for col in required_cols):
             st.error(f"matches.csv is missing required columns. Need at least: {', '.join(required_cols)}")
             return None
    except FileNotFoundError:
        st.error("Required data file 'matches.csv' not found.")
        return None
    except pd.errors.EmptyDataError:
        st.error("matches.csv is empty.")
        return None
    except Exception as e:
        st.error(f"Error loading matches.csv: {e}")
        return None

    # Data Cleaning (Comprehensive)
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', 'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Rising Pune Supergiants': 'Rising Pune Supergiants',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Gujarat Lions': 'Gujarat Lions' # Keep if present
    }
    venue_mapping = {
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium', 'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium', 'Bengaluru': 'M Chinnaswamy Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium', 'Chandigarh': 'Punjab Cricket Association IS Bindra Stadium', 'Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium', 'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium', 'Delhi': 'Arun Jaitley Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium', 'Mumbai': 'Wankhede Stadium',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium', 'Brabourne Stadium': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy', 'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy', 'Navi Mumbai': 'Dr DY Patil Sports Academy',
        'Eden Gardens, Kolkata': 'Eden Gardens', 'Kolkata': 'Eden Gardens',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium', 'Jaipur': 'Sawai Mansingh Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium', 'Chennai': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium': 'MA Chidambaram Stadium',
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium', 'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium', 'Ahmedabad': 'Narendra Modi Stadium',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium', 'Hyderabad': 'Rajiv Gandhi International Stadium', 'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium',
        'Zayed Cricket Stadium, Abu Dhabi': 'Zayed Cricket Stadium', 'Abu Dhabi': 'Zayed Cricket Stadium', 'Sheikh Zayed Stadium': 'Zayed Cricket Stadium',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium', 'Dharamsala': 'Himachal Pradesh Cricket Association Stadium', 'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium', 'Pune': 'Maharashtra Cricket Association Stadium', 'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium', 'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
         'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium', 'Dubai': 'Dubai International Cricket Stadium',
         'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium', 'Sharjah': 'Sharjah Cricket Stadium',
         'Barabati Stadium': 'Barabati Stadium', 'Cuttack': 'Barabati Stadium',
         'Holkar Cricket Stadium': 'Holkar Cricket Stadium', 'Indore': 'Holkar Cricket Stadium',
         'JSCA International Stadium Complex': 'JSCA International Stadium Complex', 'Ranchi': 'JSCA International Stadium Complex',
         'Saurashtra Cricket Association Stadium': 'Saurashtra Cricket Association Stadium', 'Rajkot': 'Saurashtra Cricket Association Stadium',
         'Vidarbha Cricket Association Stadium, Jamtha': 'Vidarbha Cricket Association Stadium', 'Nagpur': 'Vidarbha Cricket Association Stadium', 'Vidarbha Cricket Association Stadium': 'Vidarbha Cricket Association Stadium',
         'Green Park': 'Green Park', 'Kanpur': 'Green Park'
    }
    city_mapping = {'Bangalore': 'Bengaluru'}

    try:
        for col in ['team1', 'team2', 'toss_winner', 'winner']:
            if col in matches.columns:
                matches[col] = matches[col].astype(str).replace(team_name_mapping)
        if 'venue' in matches.columns:
            matches['venue'] = matches['venue'].astype(str).replace(venue_mapping)
        if 'city' in matches.columns:
            matches['city'] = matches['city'].astype(str).replace(city_mapping)
    except Exception as e:
        st.error(f"Error applying mappings: {e}")
        return None

    return matches

# --- Load Assets ---
model = load_model()
matches_df = load_data()

if model is None or matches_df is None:
    st.error("Application cannot start due to loading errors.")
    st.stop()

# --- Pre-computation for UI ---
try:
    all_teams = sorted(matches_df['team1'].dropna().unique())
    if not all_teams: raise ValueError("No teams found")
    team_encoding = {team: i for i, team in enumerate(all_teams)}
except (KeyError, ValueError) as e:
    st.error(f"Error processing teams: {e}")
    st.stop()

try:
    all_venues = sorted(matches_df['venue'].dropna().unique())
    if not all_venues: raise ValueError("No venues found")
    venue_encoding = {venue: i for i, venue in enumerate(all_venues)}
except (KeyError, ValueError) as e:
     st.error(f"Error processing venues: {e}")
     st.stop()

home_team_map = {}
if 'city' in matches_df.columns and 'team1' in matches_df.columns:
    try:
        home_team_map = matches_df.dropna(subset=['city', 'team1']).groupby('city')['team1'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).dropna().to_dict()
    except Exception as e:
        st.warning(f"Could not determine home teams: {e}")


# --- Prediction Function: 3-Over Blend (Core Features ONLY) ---
def predict_probability(state_df):
    """Predicts win probability using a smooth 3-over blend-in with core features."""
    balls_so_far = state_df['balls_so_far'].iloc[0]
    wickets_left = state_df['wickets_left'].iloc[0]
    runs_left = state_df['runs_left'].iloc[0]
    balls_left = state_df['balls_left'].iloc[0]

    required_rr = (runs_left * 6) / balls_left if balls_left > 0 else 999

    # --- Safety-net rules ---
    if runs_left <= 0: return 1.0
    if wickets_left <= 0 and runs_left > 0: return 0.0
    if balls_left <= 0 and runs_left > 0: return 0.0
    if required_rr > 40 and balls_so_far > 1: return 0.0
    # --- End ---

    # --- 1. Get the ML Model's Prediction (using only core features) ---
    predict_df = state_df.copy()
    predict_df['required_run_rate'] = required_rr

    over = balls_so_far / 6
    predict_df['phase_Middle'] = 1 if 6 < over <= 15 else 0
    predict_df['phase_Death'] = 1 if over > 15 else 0

    # --- Using only reliable, core features ---
    feature_order = [
        'batting_team', 'bowling_team', 'venue', 'balls_so_far', 'balls_left',
        'total_runs_so_far', 'runs_left', 'current_run_rate', 'required_run_rate',
        'wickets_left', 'run_rate_diff', 'is_home_team', 'phase_Middle',
        'phase_Death'
    ]

    try:
        if hasattr(model, 'get_booster') and callable(model.get_booster):
            model_features = model.get_booster().feature_names
        elif hasattr(model, 'feature_names_in_'):
             model_features = model.feature_names_in_
        else:
             model_features = feature_order # Fallback
    except Exception:
         model_features = feature_order # Fallback

    final_predict_df = pd.DataFrame(columns=model_features)

    for col in model_features:
        if col in predict_df.columns:
            final_predict_df[col] = predict_df[col].iloc[0] if isinstance(predict_df[col], pd.Series) else predict_df[col]
        else:
            final_predict_df[col] = 0 # Fill missing

    final_predict_df.replace([np.inf, -np.inf], 999, inplace=True)
    final_predict_df = final_predict_df.fillna(0)

    try:
        if len(final_predict_df) == 1:
            if final_predict_df.isnull().values.any():
                 final_predict_df = final_predict_df.fillna(0)
            # Ensure types match model (mostly float)
            for col in final_predict_df.columns:
                 try:
                     # Check if column is supposed to be numeric based on feature_order
                     if col in ['balls_so_far', 'balls_left', 'total_runs_so_far', 'runs_left',
                                'current_run_rate', 'required_run_rate', 'wickets_left', 'run_rate_diff']:
                          final_predict_df[col] = pd.to_numeric(final_predict_df[col], errors='coerce').fillna(0)
                     # Keep categorical/encoded as is (assuming they are already correct type)
                 except Exception: pass # Ignore conversion errors for non-numeric

            model_prob = model.predict_proba(final_predict_df)[0][1]
        else:
            model_prob = 0.5 # Fallback
    except Exception as e:
         st.error(f"Prediction Error: {e}")
         model_prob = 0.5 # Fallback


    # Apply Confidence Multiplier
    if wickets_left <= 3:
        confidence_multipliers = {1: 0.5, 2: 0.75, 3: 0.9}
        multiplier = confidence_multipliers.get(wickets_left, 1.0)
        model_prob *= multiplier

    # --- 2. Transition (Overs 1-3 / Balls 1-18): Blend Heuristic & Model ---
    if balls_so_far <= 18:
        initial_prob = st.session_state.get('initial_prob', 0.5)
        # Blend weight increases from 1/18 to 18/18 (1.0) over 3 overs
        model_weight = max(0, balls_so_far) / 18.0

        final_prob = ((1 - model_weight) * initial_prob) + (model_weight * model_prob)
        return np.clip(final_prob, 0.01, 0.99)


    # --- 3. Full Model Control (Over 4+ / Ball 19+): Use 100% ML Model ---
    else:
        return np.clip(model_prob, 0.01, 0.99)


# --- UI Layout ---
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Win Predictor")
st.markdown("A stable start that smoothly transitions to a live in-game model.")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Match Setup")
    if not all_teams: st.stop() # Already handled above, but good practice
    batting_team = st.selectbox("Select Batting Team", all_teams, key="sb_bat_team")

    available_bowling_teams = [t for t in all_teams if t != batting_team]
    if not available_bowling_teams: available_bowling_teams = all_teams # Allow same if only one team

    default_bowl_index = 0
    if len(available_bowling_teams) > 1: default_bowl_index = 1
    elif not available_bowling_teams: st.stop()

    bowling_team = st.selectbox("Select Bowling Team", available_bowling_teams, key="sb_bowl_team", index=default_bowl_index)

    selected_city = None
    venue = None
    possible_venues = []
    valid_cities = sorted(matches_df['city'].dropna().unique()) if 'city' in matches_df.columns else []

    if valid_cities:
        selected_city = st.selectbox("Select City", valid_cities, key="sb_city")
        if 'venue' in matches_df.columns:
            city_venues = matches_df[matches_df['city'] == selected_city]['venue'].dropna().unique()
            if city_venues.size > 0: possible_venues = sorted(city_venues)
            else:
                 possible_venues = sorted(matches_df['venue'].dropna().unique())
                 st.warning(f"No venues for {selected_city}, showing all.")
        else: st.stop() # Venue column required
    else:
        st.warning("No city data. Select venue directly.")
        if 'venue' in matches_df.columns: possible_venues = sorted(matches_df['venue'].dropna().unique())
        else: st.stop() # Venue column required

    if possible_venues:
         venue = st.selectbox("Select Venue", possible_venues, key="sb_venue")
    else: st.stop() # Need venues


    target_runs = st.number_input("Target Runs to Win", min_value=1, max_value=400, value=180, key="sb_target")

    if st.button("Start / Reset Simulation", type="primary", key="btn_start"):
        if not batting_team or not bowling_team or not venue:
             st.warning("Ensure team & venue selected.")
        elif batting_team == bowling_team and len(all_teams) > 1:
             st.warning("Teams cannot be same.")
        else:
            keys_to_keep = ['sb_bat_team', 'sb_bowl_team', 'sb_city', 'sb_venue', 'sb_target']
            keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for key in keys_to_delete: del st.session_state[key]

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
            initial_prob = np.clip(0.5 + adjustment, 0.1, 0.9)

            st.session_state.initial_prob = initial_prob
            st.session_state.probabilities = [initial_prob]
            st.session_state.overs_history = [0.0]
            st.rerun()

# --- Main Page Display ---
if st.session_state.get('simulation_started', False):
    st.header("Current Match State")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", st.session_state.get('target', 'N/A'))
    col2.metric("Runs Left", st.session_state.get('runs_left', 'N/A'))
    col3.metric("Wickets Left", st.session_state.get('wickets_left', 'N/A'))
    balls_so_far_main = st.session_state.get('balls_so_far', 0)
    col4.metric("Balls Left", 120 - balls_so_far_main)

    with st.expander("✏️ Jump to a Specific Point in the Match"):
        override_cols = st.columns(3)
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
            if new_balls_so_far < 0 or new_balls_so_far > 120 or \
               runs_left_input < 0 or runs_left_input > max_target_runs or \
               wickets_left_input < 0 or wickets_left_input > 10:
                 st.error("Invalid input values.")
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
                st.session_state.probabilities = [prob_at_jump]
                st.session_state.overs_history = [st.session_state.balls_so_far / 6]
                st.rerun()

    st.divider()
    st.header("Ball-by-Ball Input")
    input_col, plot_col = st.columns([1, 2])

    with input_col:
        if 'wickets_left' in st.session_state: # Check init
            runs_scored_val = st.selectbox("Runs on this ball:", (0, 1, 2, 3, 4, 6), key="runs_input")
            is_wicket_val = st.checkbox("Wicket on this ball?", key="wicket_input")

            if st.button("Next Ball", type="secondary", key="btn_next_ball"):
                if st.session_state.wickets_left <= 0 or st.session_state.runs_left <= 0 or st.session_state.balls_so_far >= 120:
                    st.warning("Match over or state invalid.")
                else:
                    runs_this_ball = runs_scored_val if isinstance(runs_scored_val, int) else 0
                    st.session_state.runs_left = max(0, st.session_state.runs_left - runs_this_ball)
                    st.session_state.balls_so_far += 1
                    if is_wicket_val:
                        st.session_state.wickets_left -= 1

                    balls_left = 120 - st.session_state.balls_so_far
                    target = st.session_state.get('target', st.session_state.runs_left)
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

                    win_prob = predict_probability(state_df)

                    if 'probabilities' not in st.session_state: st.session_state.probabilities = [st.session_state.initial_prob]
                    if 'overs_history' not in st.session_state: st.session_state.overs_history = [0.0]

                    st.session_state.probabilities.append(win_prob)
                    st.session_state.overs_history.append(st.session_state.balls_so_far / 6)
                    st.rerun()
        else:
            st.info("Simulation not initialized.")


    with plot_col:
        st.subheader("Win Probability Chart")
        if st.session_state.get('probabilities') and isinstance(st.session_state.probabilities, list) and \
           st.session_state.get('overs_history') and isinstance(st.session_state.overs_history, list):
            try:
                probs = st.session_state.probabilities
                overs = st.session_state.overs_history
                min_len = min(len(overs), len(probs))

                if min_len < 2: # Need at least 2 points for a line
                    st.info("Need more data points to plot chart.")
                else:
                    overs_hist = overs[:min_len]
                    numeric_probs = [p for p in probs[:min_len] if isinstance(p, (int, float))]

                    if len(numeric_probs) != min_len:
                        st.warning("Skipping plot due to non-numeric probability values.")
                    else:
                        bat_probs = np.array(numeric_probs)
                        if np.isnan(bat_probs).any() or np.isinf(bat_probs).any():
                             st.warning("Skipping plot due to invalid probability values (NaN/inf).")
                        else:
                            bowl_probs = 1 - bat_probs
                            fig, ax = plt.subplots(figsize=(10, 6))
                            batting_label = st.session_state.get('batting_team', 'Batting Team')
                            bowling_label = st.session_state.get('bowling_team', 'Bowling Team')

                            ax.plot(overs_hist, bat_probs, label=batting_label, color="#0072B2", linewidth=2.5, marker='o', markersize=4) # Smaller markers
                            ax.plot(overs_hist, bowl_probs, label=bowling_label, color="#D55E00", linewidth=2.5, marker='o', markersize=4) # Smaller markers

                            ax.axhline(0.5, linestyle="--", color="grey", alpha=0.8)
                            ax.set_xlabel("Overs"); ax.set_ylabel("Win Probability"); ax.set_title("Live Win Probability", fontsize=16)
                            ax.set_ylim(0, 1); ax.set_xlim(min(overs_hist or [0]) - 0.5, 20);
                            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                            ax.legend()
                            st.pyplot(fig)

                            if bat_probs.size > 0:
                                final_prob_val = bat_probs[-1] * 100
                                st.metric(label=f"**{batting_label}'s Current Win Probability**", value=f"{final_prob_val:.2f}%")
                            else:
                                st.metric(label=f"**{batting_label}'s Current Win Probability**", value="N/A")

            except Exception as e:
                st.error(f"Error plotting chart: {e}")
        else:
             st.info("Waiting for simulation data.")


elif not st.session_state.get('simulation_started', False):
    st.info("Setup a match in the sidebar and click 'Start / Reset Simulation' to begin.")
