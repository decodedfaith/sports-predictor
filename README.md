# sports-predictor

This project scrapes soccer match data from Flashscore and uses a machine learning model to predict match outcomes.

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Set up your `.env` file with your Apify API token.

## Usage

1. Run `app.py` to scrape data and save it to `flashscore_data.csv`.
2. Run `model.py` to train and evaluate the RandomForest model.
