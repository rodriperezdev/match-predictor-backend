from data_collection import collect_all_data
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('FOOTBALL_API_KEY')
    if not api_key:
        raise ValueError("FOOTBALL_API_KEY not found in environment")
    
    print("🚀 Starting data collection...")
    matches_df = collect_all_data(api_key, seasons=[2023, 2024])
    print("\n✅ Data collection complete!")