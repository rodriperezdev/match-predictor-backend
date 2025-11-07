import requests
import time
import re
from bs4 import BeautifulSoup  # type: ignore
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

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

class OddsCollector:
    """Collects betting odds from The Odds API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def get_odds_for_league(self, sport='soccer_argentina_primera_division'):
        """Get current odds for Argentine Primera División"""
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us,uk',
            'markets': 'h2h',  # Head to head (match winner)
            'oddsFormat': 'decimal'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                remaining = response.headers.get('x-requests-remaining', '?')
                print(f"✓ Odds API request | Remaining: {remaining}")
                return response.json()
            else:
                print(f"✗ Odds API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Odds API exception: {e}")
            return None
    
    def save_historical_odds(self, output_file='data/odds_history.csv'):
        """Save current odds to CSV for historical tracking"""
        odds_data = self.get_odds_for_league()
        
        if not odds_data:
            return
        
        odds_records = []
        for match in odds_data:
            home_team = match['home_team']
            away_team = match['away_team']
            commence_time = match['commence_time']
            
            # Get average odds from bookmakers
            if match.get('bookmakers'):
                home_odds = []
                draw_odds = []
                away_odds = []
                
                for bookmaker in match['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == home_team:
                                    home_odds.append(outcome['price'])
                                elif outcome['name'] == away_team:
                                    away_odds.append(outcome['price'])
                                elif outcome['name'] == 'Draw':
                                    draw_odds.append(outcome['price'])
                
                if home_odds and away_odds:
                    odds_records.append({
                        'date': commence_time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': np.mean(home_odds),
                        'draw_odds': np.mean(draw_odds) if draw_odds else None,
                        'away_odds': np.mean(away_odds)
                    })
        
        if odds_records:
            df = pd.DataFrame(odds_records)
            
            # Append to existing or create new
            if os.path.exists(output_file):
                existing = pd.read_csv(output_file)
                df = pd.concat([existing, df]).drop_duplicates()
            
            os.makedirs('data', exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"✓ Saved {len(odds_records)} odds records")
        else:
            print("⚠️ No odds records to save")

# ESPN team ID mapping for Argentine Primera División (Liga Profesional)
ESPN_TEAM_IDS = {
    'River Plate': '16',
    'Boca Juniors': '15',
    'Racing Club': '17',
    'Independiente': '18',
    'San Lorenzo': '19',
    'Estudiantes L.P.': '20',
    'Estudiantes de La Plata': '20',
    'Belgrano Cordoba': '4',
    'Belgrano (Córdoba)': '4',
    'Rosario Central': '22',
    'Colon Santa Fe': '23',
    'Barracas Central': '24',
    'Tigre': '25',
    'Newell\'s Old Boys': '26',
    'Talleres (Córdoba)': '27',
    'Defensa y Justicia': '28',
    'Central Córdoba (Santiago del Estero)': '29',
    'Aldosivi': '30',
    'Banfield': '31',
    'San Martín (San Juan)': '32',
    # Add more teams as needed
}

def scrape_espn_recent_matches(team_name: str, limit: int = 5):
    """Scrape recent matches from ESPN Argentina for a team - Liga Profesional only"""
    # Try to find team ID
    team_id = None
    matched_team = None
    for team, espn_id in ESPN_TEAM_IDS.items():
        if team_name.lower() in team.lower() or team.lower() in team_name.lower():
            team_id = espn_id
            matched_team = team
            break
    
    if not team_id:
        print(f"⚠️ Team ID not found for {team_name}, trying fallback URL")
        return None
    
    # Use the correct URL format for Liga Profesional (ARG.1)
    url = f"https://www.espn.com.ar/futbol/equipo/resultados/_/id/{team_id}/liga/ARG.1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        time.sleep(2)  # Be respectful
        
        if response.status_code != 200:
            print(f"⚠️ ESPN request failed for {team_name}: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        matches = []
        
        # ESPN uses tables for match results
        # Find all table rows that contain match data
        tables = soup.find_all('table')
        match_rows = []
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                # Skip header rows
                if row.find('th'):
                    continue
                # Look for rows with match data (should have date, teams, score)
                text = row.get_text()
                # Check if it looks like a match row (has date pattern or score pattern)
                if re.search(r'\d+\s*[-–]\s*\d+', text) or 'Finalizado' in text:
                    match_rows.append(row)
        
        # Also try finding by class names that ESPN might use
        if not match_rows:
            match_rows = soup.find_all('tr', class_=lambda x: x and ('match' in str(x).lower() or 'game' in str(x).lower()))
        
        # Filter for Liga Profesional only
        filtered_rows = []
        for row in match_rows:
            row_text = row.get_text()
            # Only include Liga Profesional matches
            if 'Liga Profesional' in row_text or 'Liga Profesional de Fútbol' in row_text:
                filtered_rows.append(row)
        
        if not filtered_rows:
            # If no Liga Profesional filter found, use all rows
            filtered_rows = match_rows
        
        for container in filtered_rows[:limit]:
            try:
                # Extract all cells/columns from the row
                cells = container.find_all(['td', 'div'])
                if len(cells) < 3:
                    # Try getting text directly
                    full_text = container.get_text(separator=' | ', strip=True)
                    cells = [full_text]
                
                # ESPN structure: FECHA | Partido | Resultado | COMPETENCIA
                # Partido column typically has: "Team1 X - Y Team2" or "Team1\nX - Y\nTeam2"
                full_text = container.get_text(separator='\n', strip=True)
                
                # Extract score (most reliable)
                score_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', full_text)
                if not score_match:
                    continue
                
                score1 = int(score_match.group(1))
                score2 = int(score_match.group(2))
                
                # Find the "Partido" column - it contains both team names
                # ESPN format: "Team1\nX - Y\nTeam2" or "Team1 X - Y Team2"
                team_name_variants = [
                    team_name,
                    matched_team if matched_team else team_name,
                    'Belgrano (Córdoba)' if 'belgrano' in team_name.lower() else team_name,
                    'Belgrano Cordoba' if 'belgrano' in team_name.lower() else team_name,
                ]
                
                team_name_found = None
                for variant in team_name_variants:
                    if variant and variant.lower() in full_text.lower():
                        team_name_found = variant
                        break
                
                if not team_name_found:
                    continue
                
                # Split text by newlines to find team positions
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                
                # Find which line has the team name
                team_line_idx = -1
                for idx, line in enumerate(lines):
                    if team_name_found.lower() in line.lower():
                        team_line_idx = idx
                        break
                
                # Find score line
                score_line_idx = -1
                for idx, line in enumerate(lines):
                    if re.search(r'\d+\s*[-–]\s*\d+', line):
                        score_line_idx = idx
                        break
                
                # Determine home/away: In ESPN, first team listed is usually home
                # If our team is on the line before score or first in the match section, it's home
                is_home = False
                opponent = 'Unknown'
                
                # Try to find opponent from known teams
                for known_team in ESPN_TEAM_IDS.keys():
                    if known_team.lower() != team_name_found.lower():
                        # Check if opponent appears in the text
                        if known_team.lower() in full_text.lower():
                            # Check if it's in a different line or position
                            for line in lines:
                                if known_team.lower() in line.lower() and team_name_found.lower() not in line.lower():
                                    opponent = known_team
                                    # Determine home/away: if our team comes first, we're home
                                    team_pos = full_text.lower().find(team_name_found.lower())
                                    opp_pos = full_text.lower().find(known_team.lower())
                                    is_home = team_pos < opp_pos
                                    break
                            if opponent != 'Unknown':
                                break
                
                # If opponent not found in known teams, try to extract from structure
                if opponent == 'Unknown':
                    # Look for text that's not our team name and not the score
                    for line in lines:
                        line_clean = line.strip()
                        if (line_clean and 
                            team_name_found.lower() not in line_clean.lower() and
                            not re.search(r'\d+\s*[-–]\s*\d+', line_clean) and
                            'Finalizado' not in line_clean and
                            len(line_clean) > 3):
                            opponent = line_clean
                            # Determine home/away
                            team_pos = full_text.lower().find(team_name_found.lower())
                            opp_pos = full_text.lower().find(opponent.lower())
                            is_home = team_pos < opp_pos if opp_pos != -1 else False
                            break
                
                # Assign goals: ESPN shows "Team1 X - Y Team2" where X is Team1's goals
                if is_home:
                    team_goals = score1
                    opponent_goals = score2
                else:
                    team_goals = score2
                    opponent_goals = score1
                
                # Determine result
                if team_goals > opponent_goals:
                    result = 'W'
                elif team_goals < opponent_goals:
                    result = 'L'
                else:
                    result = 'D'
                
                # Extract date if available
                date_elem = container.find(['span', 'td'], class_=lambda x: x and 'date' in x.lower())
                date_str = date_elem.get_text(strip=True) if date_elem else ''
                
                matches.append({
                    'opponent': opponent,
                    'is_home': is_home,
                    'team_goals': team_goals,
                    'opponent_goals': opponent_goals,
                    'result': result,
                    'date': date_str
                })
            except Exception as e:
                continue
        
        if matches:
            print(f"✓ Scraped {len(matches)} recent matches from ESPN for {team_name}")
            return matches
        else:
            print(f"⚠️ No matches found on ESPN for {team_name}")
            return None
            
    except Exception as e:
        print(f"✗ ESPN scraping error for {team_name}: {e}")
        return None

def scrape_fbref_stats():
    """Scrape comprehensive team stats from FBref"""
    url = "https://fbref.com/en/comps/21/Primera-Division-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        time.sleep(3)
        
        if response.status_code != 200:
            print("⚠️ FBref request failed")
            return None
        
        # Parse tables
        tables = pd.read_html(response.content)
        
        if not tables:
            print("⚠️ No tables found")
            return None
        
        # Main stats table (usually first)
        stats_df = tables[0]
        
        # Flatten multi-level columns if present
        if isinstance(stats_df.columns, pd.MultiIndex):
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        
        # Clean team names
        if 'Squad' in stats_df.columns:
            stats_df = stats_df.rename(columns={'Squad': 'team'})
        elif any('Squad' in str(col) for col in stats_df.columns):
            squad_col = [col for col in stats_df.columns if 'Squad' in str(col)][0]
            stats_df = stats_df.rename(columns={squad_col: 'team'})
        
        # Select relevant columns (adjust based on actual FBref structure)
        relevant_cols = ['team']
        optional_cols = [
            'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts',  # Basic
            'xG', 'xGA', 'xGD',  # Expected goals
            'Poss', 'Possession',  # Possession
            'Sh', 'SoT', 'Shots', 'SoT%',  # Shots
            'Ast', 'PK', 'PKatt'  # Other
        ]
        
        for col in optional_cols:
            matching = [c for c in stats_df.columns if col in str(c)]
            if matching:
                relevant_cols.append(matching[0])
        
        stats_df = stats_df[relevant_cols]
        
        # Convert numeric columns
        for col in stats_df.columns:
            if col != 'team':
                stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        os.makedirs('data', exist_ok=True)
        stats_df.to_csv('data/fbref_stats.csv', index=False)
        print(f"✓ Scraped stats for {len(stats_df)} teams from FBref")
        
        return stats_df
        
    except Exception as e:
        print(f"✗ FBref scraping error: {e}")
        return None

def collect_all_data(api_key, odds_api_key, seasons=[2023, 2024]):
    """Main collection function"""
    LEAGUE_ID = 128  # Argentine Primera División
    
    # Collect match fixtures
    collector = FootballDataCollector(api_key)
    all_fixtures = []
    
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
    
    # Collect betting odds
    print("\n💰 Collecting betting odds...")
    odds_collector = OddsCollector(odds_api_key)
    odds_collector.save_historical_odds()
    
    # Scrape FBref stats
    print("\n📈 Scraping FBref advanced stats...")
    fbref_stats = scrape_fbref_stats()
    
    # Save everything
    os.makedirs('data', exist_ok=True)
    matches_df.to_csv('data/matches.csv', index=False)
    
    print(f"\n✓ Collected {len(matches_df)} matches")
    print(f"✓ Data saved to data/")
    
    return matches_df, fbref_stats