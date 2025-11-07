import requests
import time
import json
from datetime import datetime
from bs4 import BeautifulSoup  # type: ignore
import pandas as pd
import os

class FootballDataCollector:
    """Collects data from API-Football with rate limiting"""
    
    def __init__(self, api_key, max_requests=80):
        self.api_key = api_key
        self.headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': api_key
        }
        self.base_url = "https://v3.football.api-sports.io"
        self.request_count = 0
        self.max_requests = max_requests
        
    def make_request(self, endpoint, params):
        """Make API request with rate limiting"""
        if self.request_count >= self.max_requests:
            print(f"⚠️ Reached daily limit")
            return None
            
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.request_count += 1
            
            if response.status_code == 200:
                remaining = response.headers.get('x-ratelimit-requests-remaining', '?')
                print(f"✓ Request {self.request_count}/{self.max_requests} | Remaining: {remaining}")
                time.sleep(6)  # 10 req/min = 1 per 6 sec
                return response.json()
            else:
                print(f"✗ Error {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            return None
    
    def get_fixtures(self, league_id, season):
        """Get all fixtures for a season"""
        return self.make_request('fixtures', {'league': league_id, 'season': season})
    
    def get_standings(self, league_id, season):
        """Get league standings"""
        return self.make_request('standings', {'league': league_id, 'season': season})

def scrape_fbref_stats():
    """Scrape team stats from FBref"""
    url = "https://fbref.com/en/comps/21/Primera-Division-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        time.sleep(3)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if tables:
                df = pd.read_html(str(tables[0]))[0]
                print(f"✓ Scraped {len(df)} teams from FBref")
                return df
        
        print("⚠️ FBref scraping failed, using fallback")
        return None
        
    except Exception as e:
        print(f"✗ FBref error: {str(e)}")
        return None

def collect_all_data(api_key, seasons=[2023, 2024]):
    """Main collection function"""
    LEAGUE_ID = 128  # Argentine Primera División
    
    collector = FootballDataCollector(api_key)
    all_fixtures = []
    
    # Collect fixtures
    for season in seasons:
        print(f"\n📊 Collecting season {season}...")
        data = collector.get_fixtures(LEAGUE_ID, season)
        if data and 'response' in data:
            all_fixtures.extend(data['response'])
    
    # Process fixtures
    matches = []
    for fixture in all_fixtures:
        if fixture['fixture']['status']['short'] != 'FT':
            continue
            
        match = {
            'date': fixture['fixture']['date'],
            'season': fixture['league']['season'],
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'home_team_id': fixture['teams']['home']['id'],
            'away_team_id': fixture['teams']['away']['id'],
            'home_goals': fixture['goals']['home'],
            'away_goals': fixture['goals']['away'],
        }
        
        # Outcome: 2=Home Win, 1=Draw, 0=Away Win
        if fixture['teams']['home']['winner']:
            match['outcome'] = 2
        elif fixture['teams']['away']['winner']:
            match['outcome'] = 0
        else:
            match['outcome'] = 1
            
        matches.append(match)
    
    matches_df = pd.DataFrame(matches)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df = matches_df.sort_values('date').reset_index(drop=True)
    
    # Save
    os.makedirs('data', exist_ok=True)
    matches_df.to_csv('data/matches.csv', index=False)
    
    print(f"\n✓ Collected {len(matches_df)} matches")
    print(f"✓ Saved to data/matches.csv")
    
    return matches_df