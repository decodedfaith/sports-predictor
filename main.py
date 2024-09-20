import os
import pandas as pd  # Import pandas
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables from .env file
load_dotenv()
print("Environment variables loaded.")

# Retrieve the APIFY_TOKEN from the environment
apify_token = os.getenv('APIFY_TOKEN')

print(f"APIFY_TOKEN: {apify_token}")  # Add this line after loading the token


# Check if the token is available
if not apify_token:
    raise ValueError("APIFY_TOKEN not found in environment variables. Ensure it's set in your .env file.")

# Initialize the ApifyClient with the token
client = ApifyClient(apify_token)
print("ssssssssssssss")
# Input configuration for the Flashscore scraper
run_input = {
    "inputURL": [
        {"url": "https://www.flashscore.com/soccer/england/premier-league/results/"}
    ],
    "proxyConfiguration": {
        "useApifyProxy": True  # Use Apify's proxy
    }
}
print("jjjjjjjjj")

# Trigger the scraper actor and retrieve results
try:
    run = client.actor("tomas_jindra/flashscore-scraper").call(run_input=run_input)
    print("lwwwwlwlww")
    dataset_items = client.dataset(run['defaultDatasetId']).list_items().items
    print("Scraped Data:", dataset_items)

    # Save scraped data to CSV
    df = pd.DataFrame(dataset_items)
    df.to_csv('flashscore_data.csv', index=False)
    print("Data saved to flashscore_data.csv")

except Exception as e:
    print(f"Error during scraping: {e}")
