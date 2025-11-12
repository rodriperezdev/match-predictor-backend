from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Dict, List
import os
from model_training import calculate_base_features
from data_collection import scrape_espn_recent_matches

app = FastAPI(title="Football Match Predictor API")

# CORS Configuration
# Allow specific origins from environment variable, or default to all for development
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    # Split comma-separated origins from environment variable
    cors_origins: List[str] = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default: allow all origins (for development)
    # Use "*" alone - FastAPI will allow all origins
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
scaler = None
feature_cols = None
matches_df = None

@app.on_event("startup")
def load_model():
    global model, scaler, feature_cols, matches_df
    
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/features.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        matches_df = pd.read_csv('data/matches.csv')
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    confidence: float

def calculate_features(home_team, away_team):
    """Calculate features for prediction"""
    return calculate_base_features(matches_df, home_team, away_team)

@app.get("/")
def root():
    return {"message": "Football Match Predictor API", "status": "running"}

@app.get("/teams")
def get_teams():
    """Get list of all teams"""
    if matches_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    teams = sorted(matches_df['home_team'].unique().tolist())
    return {"teams": teams}

@app.post("/predict", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    """Predict match outcome"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Calculate features
        features = calculate_features(request.home_team, request.away_team)
        
        # Create dataframe
        X = pd.DataFrame([features])[feature_cols]
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        return PredictionResponse(
            prediction=outcome_map[prediction],
            probabilities={
                'away_win': float(probabilities[0]),
                'draw': float(probabilities[1]),
                'home_win': float(probabilities[2])
            },
            confidence=float(probabilities[prediction])
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get dataset statistics"""
    if matches_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "total_matches": len(matches_df),
        "seasons": sorted(matches_df['season'].unique().tolist()),
        "home_wins": int((matches_df['outcome'] == 2).sum()),
        "draws": int((matches_df['outcome'] == 1).sum()),
        "away_wins": int((matches_df['outcome'] == 0).sum()),
    }

@app.get("/recent-matches/{team_name}")
def get_recent_matches(team_name: str, limit: int = 5, use_espn: bool = True):
    """Get recent matches for a team - tries ESPN first, falls back to local data"""
    matches_list = []
    
    # Try ESPN scraping first for more recent data
    if use_espn:
        try:
            espn_matches = scrape_espn_recent_matches(team_name, limit)
            if espn_matches and len(espn_matches) > 0:
                # Convert ESPN format to API format
                for match in espn_matches[:limit]:
                    matches_list.append({
                        "date": match.get('date', ''),
                        "opponent": match.get('opponent', 'Unknown'),
                        "is_home": bool(match.get('is_home', False)),  # Explicitly boolean
                        "team_goals": int(match.get('team_goals', 0)),
                        "opponent_goals": int(match.get('opponent_goals', 0)),
                        "result": match.get('result', 'D'),
                    })
                
                if matches_list:
                    return {"matches": matches_list}
        except Exception as e:
            print(f"[WARNING] ESPN scraping failed, falling back to local data: {e}")
    
    # Fallback to local data if ESPN fails or is disabled
    if matches_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Clean team name for matching
    team_name_clean = str(team_name).strip()
    
    # Find matches where team is home or away
    def team_matches(row):
        home = str(row['home_team']).strip() if pd.notna(row['home_team']) else ''
        away = str(row['away_team']).strip() if pd.notna(row['away_team']) else ''
        return home == team_name_clean or away == team_name_clean
    
    team_matches_df = matches_df[
        matches_df.apply(team_matches, axis=1)
    ].copy()
    
    # Filter out matches with missing critical data
    team_matches_df = team_matches_df[
        team_matches_df['home_goals'].notna() & 
        team_matches_df['away_goals'].notna() &
        team_matches_df['outcome'].notna() &
        team_matches_df['date'].notna()
    ]
    
    # Sort by date descending and get most recent
    team_matches_df = team_matches_df.sort_values('date', ascending=False).head(limit)
    
    if len(team_matches_df) == 0:
        return {"matches": []}
    
    for _, match in team_matches_df.iterrows():
        # Clean team names
        home_team = str(match['home_team']).strip() if pd.notna(match['home_team']) else ''
        away_team = str(match['away_team']).strip() if pd.notna(match['away_team']) else ''
        
        # Determine if team was home or away - CRITICAL: exact match
        is_home = (home_team == team_name_clean)
        is_away = (away_team == team_name_clean)
        
        # Safety check
        if not is_home and not is_away:
            continue
        
        # Get opponent and goals
        if is_home:
            opponent = away_team
            team_goals = int(match['home_goals']) if pd.notna(match['home_goals']) else 0
            opponent_goals = int(match['away_goals']) if pd.notna(match['away_goals']) else 0
        else:  # is_away
            opponent = home_team
            team_goals = int(match['away_goals']) if pd.notna(match['away_goals']) else 0
            opponent_goals = int(match['home_goals']) if pd.notna(match['home_goals']) else 0
        
        # Determine result
        outcome = match['outcome']
        if pd.isna(outcome):
            if team_goals > opponent_goals:
                result = 'W'
            elif team_goals < opponent_goals:
                result = 'L'
            else:
                result = 'D'
        elif is_home:
            if outcome == 2:
                result = 'W'
            elif outcome == 0:
                result = 'L'
            else:
                result = 'D'
        else:  # is_away
            if outcome == 0:
                result = 'W'
            elif outcome == 2:
                result = 'L'
            else:
                result = 'D'
        
        # Format date
        try:
            date_str = match['date'].strftime('%Y-%m-%d') if pd.notna(match['date']) else ''
        except:
            date_str = str(match['date']) if pd.notna(match['date']) else ''
        
        matches_list.append({
            "date": date_str,
            "opponent": opponent,
            "is_home": bool(is_home),  # CRITICAL: Explicit boolean conversion
            "team_goals": team_goals,
            "opponent_goals": opponent_goals,
            "result": result,
        })
    
    return {"matches": matches_list}