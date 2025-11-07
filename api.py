from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Dict
from model_training import calculate_recent_form, calculate_goal_difference, calculate_h2h

app = FastAPI(title="Football Match Predictor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    features = {
        'home_form_5': calculate_recent_form(matches_df, home_team, 5, True),
        'away_form_5': calculate_recent_form(matches_df, away_team, 5, False),
        'home_form_10': calculate_recent_form(matches_df, home_team, 10, True),
        'away_form_10': calculate_recent_form(matches_df, away_team, 10, False),
        'home_gd': calculate_goal_difference(matches_df, home_team),
        'away_gd': calculate_goal_difference(matches_df, away_team),
        'h2h_home_advantage': calculate_h2h(matches_df, home_team, away_team),
        'home_advantage': 1,
    }
    return features

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
        "seasons": matches_df['season'].unique().tolist(),
        "home_wins": int((matches_df['outcome'] == 2).sum()),
        "draws": int((matches_df['outcome'] == 1).sum()),
        "away_wins": int((matches_df['outcome'] == 0).sum()),
    }