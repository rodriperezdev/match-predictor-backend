import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
import xgboost as xgb  # type: ignore
import os

def calculate_recent_form(df, team, n=5, as_home=True):
    """Calculate recent form (0-1 scale)"""
    if as_home:
        recent = df[df['home_team'] == team].tail(n)
        points = recent['outcome'].map({2: 3, 1: 1, 0: 0}).sum()
    else:
        recent = df[df['away_team'] == team].tail(n)
        points = recent['outcome'].map({0: 3, 1: 1, 2: 0}).sum()
    
    return points / (n * 3) if n > 0 else 0.5

def calculate_goal_difference(df, team, last_n=10):
    """Calculate goal difference"""
    home = df[df['home_team'] == team].tail(last_n)
    away = df[df['away_team'] == team].tail(last_n)
    
    home_gd = (home['home_goals'] - home['away_goals']).sum()
    away_gd = (away['away_goals'] - away['home_goals']).sum()
    
    return home_gd + away_gd

def calculate_h2h(df, team1, team2, last_n=5):
    """Head-to-head record"""
    h2h = df[
        ((df['home_team'] == team1) & (df['away_team'] == team2)) |
        ((df['home_team'] == team2) & (df['away_team'] == team1))
    ].tail(last_n)
    
    if len(h2h) == 0:
        return 0.5
    
    team1_wins = len(h2h[
        ((h2h['home_team'] == team1) & (h2h['outcome'] == 2)) |
        ((h2h['away_team'] == team1) & (h2h['outcome'] == 0))
    ])
    
    return team1_wins / len(h2h)

def engineer_features(matches_df):
    """Create features for each match"""
    features_list = []
    
    for idx, match in matches_df.iterrows():
        historical = matches_df.iloc[:idx]
        
        if len(historical) < 10:
            continue
        
        features = {
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'outcome': match['outcome'],
            'home_form_5': calculate_recent_form(historical, match['home_team'], 5, True),
            'away_form_5': calculate_recent_form(historical, match['away_team'], 5, False),
            'home_form_10': calculate_recent_form(historical, match['home_team'], 10, True),
            'away_form_10': calculate_recent_form(historical, match['away_team'], 10, False),
            'home_gd': calculate_goal_difference(historical, match['home_team']),
            'away_gd': calculate_goal_difference(historical, match['away_team']),
            'h2h_home_advantage': calculate_h2h(historical, match['home_team'], match['away_team']),
            'home_advantage': 1,
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def train_models(features_df):
    """Train and evaluate multiple models"""
    X = features_df.drop(['outcome', 'home_team', 'away_team'], axis=1)
    y = features_df['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    }
    
    results = {}
    best_accuracy = 0
    best_model = None
    best_name = None
    
    print("\n🤖 Training models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best: {best_name} ({best_accuracy:.4f})")
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("✓ Models saved to models/")
    
    return best_model, scaler, X.columns.tolist(), results

def train_pipeline():
    """Complete training pipeline"""
    print("\n📊 Loading data...")
    matches_df = pd.read_csv('data/matches.csv')
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    print(f"✓ Loaded {len(matches_df)} matches")
    
    print("\n🔧 Engineering features...")
    features_df = engineer_features(matches_df)
    features_df.to_csv('data/features.csv', index=False)
    print(f"✓ Created {len(features_df)} feature records")
    
    print("\n🤖 Training models...")
    model, scaler, features, results = train_models(features_df)
    
    return model, scaler, features, results