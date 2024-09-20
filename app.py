import streamlit as st
import pandas as pd
import joblib
import ast

# Load the model and data
model = joblib.load('model.pkl')
data = pd.read_csv('flashscore_data.csv')

# Function to extract values from string formatted as dictionaries
def extract_value(column_value, key):
    try:
        return float(ast.literal_eval(column_value)[key].replace('%', ''))  # Remove percentage sign if present
    except (ValueError, KeyError, SyntaxError):
        return 0

# Parse required columns if not already parsed
required_columns = [
    ('Expected Goals (xG)', 'homeTeamValue', 'xG_home'),
    ('Expected Goals (xG)', 'awayTeamValue', 'xG_away'),
    ('Ball Possession', 'homeTeamValue', 'possession_home'),
    ('Ball Possession', 'awayTeamValue', 'possession_away'),
    ('Goal Attempts', 'homeTeamValue', 'goal_attempts_home'),
    ('Goal Attempts', 'awayTeamValue', 'goal_attempts_away'),
    ('Shots on Goal', 'homeTeamValue', 'shots_on_goal_home'),
    ('Shots on Goal', 'awayTeamValue', 'shots_on_goal_away'),
    ('Total Passes', 'homeTeamValue', 'total_passes_home'),
    ('Total Passes', 'awayTeamValue', 'total_passes_away'),
    ('Fouls', 'homeTeamValue', 'fouls_home'),
    ('Fouls', 'awayTeamValue', 'fouls_away'),
    ('Yellow Cards', 'homeTeamValue', 'yellow_cards_home'),
    ('Yellow Cards', 'awayTeamValue', 'yellow_cards_away'),
    ('Red Cards', 'homeTeamValue', 'red_cards_home'),
    ('Red Cards', 'awayTeamValue', 'red_cards_away'),
]

for column_name, key, new_column in required_columns:
    if new_column not in data.columns:
        data[new_column] = data[column_name].apply(lambda x: extract_value(x, key))

# Debugging: Show available columns
st.write("Columns in data:", data.columns)

# Streamlit app UI
st.title("Sports Match Outcome Predictor")

# User selects two teams from dropdowns
team_a = st.selectbox("Select Team A", data['homeTeamName'].unique())
team_b = st.selectbox("Select Team B", data['awayTeamName'].unique())

# Predict the match outcome
if st.button("Predict"):
    try:
        # Get stats for selected teams
        team_a_stats = data[data['homeTeamName'] == team_a].iloc[0]
        team_b_stats = data[data['awayTeamName'] == team_b].iloc[0]
        
        # Ensure that all necessary columns are available for prediction
        features = [
            team_a_stats['xG_home'], team_b_stats['xG_away'],
            team_a_stats['possession_home'], team_b_stats['possession_away'],
            team_a_stats['goal_attempts_home'], team_b_stats['goal_attempts_away'],
            team_a_stats['shots_on_goal_home'], team_b_stats['shots_on_goal_away'],
            team_a_stats['total_passes_home'], team_b_stats['total_passes_away'],
            team_a_stats['fouls_home'], team_b_stats['fouls_away'],
            team_a_stats['yellow_cards_home'], team_b_stats['yellow_cards_away'],
            team_a_stats['red_cards_home'], team_b_stats['red_cards_away']
        ]
        
        prediction = model.predict([features])
        result = "Team A wins" if prediction[0] == 1 else "Team B wins"
        st.write(f"Predicted Outcome: {result}")
    
    except KeyError as e:
        st.error(f"KeyError: {e} - Check if the necessary columns exist.")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
