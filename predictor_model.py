import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import ast
import joblib  # Import joblib to save the model

# Load the dataset
data = pd.read_csv('flashscore_data.csv')
print("Dataset loaded.")

# Function to extract values from string formatted as dictionaries
def extract_value(column_value, key):
    try:
        return float(ast.literal_eval(column_value)[key])
    except (ValueError, KeyError, SyntaxError):
        return 0

# Extract features
data['xG_home'] = data['Expected Goals (xG)'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['xG_away'] = data['Expected Goals (xG)'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['possession_home'] = data['Ball Possession'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['possession_away'] = data['Ball Possession'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['goal_attempts_home'] = data['Goal Attempts'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['goal_attempts_away'] = data['Goal Attempts'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['shots_on_goal_home'] = data['Shots on Goal'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['shots_on_goal_away'] = data['Shots on Goal'].apply(lambda x: extract_value(x, 'awayTeamValue'))

# Extract additional features
data['total_passes_home'] = data['Total Passes'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['total_passes_away'] = data['Total Passes'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['fouls_home'] = data['Fouls'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['fouls_away'] = data['Fouls'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['yellow_cards_home'] = data['Yellow Cards'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['yellow_cards_away'] = data['Yellow Cards'].apply(lambda x: extract_value(x, 'awayTeamValue'))
data['red_cards_home'] = data['Red Cards'].apply(lambda x: extract_value(x, 'homeTeamValue'))
data['red_cards_away'] = data['Red Cards'].apply(lambda x: extract_value(x, 'awayTeamValue'))

# Create target variable
data['outcome'] = data.apply(lambda row: 1 if row['homeTeamScore'] > row['awayTeamScore'] else 0, axis=1)

# Select features for modeling
X = data[['xG_home', 'xG_away', 'possession_home', 'possession_away', 
           'goal_attempts_home', 'goal_attempts_away', 'shots_on_goal_home', 
           'shots_on_goal_away', 'total_passes_home', 'total_passes_away', 
           'fouls_home', 'fouls_away', 'yellow_cards_home', 'yellow_cards_away',
           'red_cards_home', 'red_cards_away']]
y = data['outcome']

# Handle missing values
X = X.fillna(0).copy()
X = X.apply(pd.to_numeric, errors='coerce')

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and tune the model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(best_model, 'model.pkl')
print("Model saved as 'model.pkl'")
