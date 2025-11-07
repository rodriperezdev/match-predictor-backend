import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  # type: ignore
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

def calculate_base_features(historical_df, home_team, away_team):
    """Calculate base features for a match - shared between training and prediction"""
    # Calculate relative strength difference (home - away)
    home_form_5 = calculate_recent_form(historical_df, home_team, 5, True)
    away_form_5 = calculate_recent_form(historical_df, away_team, 5, False)
    home_form_10 = calculate_recent_form(historical_df, home_team, 10, True)
    away_form_10 = calculate_recent_form(historical_df, away_team, 10, False)
    home_gd = calculate_goal_difference(historical_df, home_team)
    away_gd = calculate_goal_difference(historical_df, away_team)
    
    return {
        'home_form_5': home_form_5,
        'away_form_5': away_form_5,
        'home_form_10': home_form_10,
        'away_form_10': away_form_10,
        'home_gd': home_gd,
        'away_gd': away_gd,
        'form_diff_5': home_form_5 - away_form_5,  # Relative form difference
        'form_diff_10': home_form_10 - away_form_10,  # Relative form difference
        'gd_diff': home_gd - away_gd,  # Goal difference advantage
        'h2h_home_advantage': calculate_h2h(historical_df, home_team, away_team),
        'home_advantage': 0.15,  # Small home advantage (reduced from 1.0)
    }

def merge_odds_data(matches_df, odds_df):
    """Merge betting odds with match data"""
    if odds_df is None or len(odds_df) == 0:
        print("⚠️ No odds data available")
        return matches_df
    
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    
    # Merge on date and team names
    merged = matches_df.merge(
        odds_df,
        on=['date', 'home_team', 'away_team'],
        how='left'
    )
    
    return merged

def merge_fbref_stats(matches_df, fbref_df):
    """Add FBref advanced stats to matches"""
    if fbref_df is None or len(fbref_df) == 0:
        print("⚠️ No FBref data available")
        return matches_df
    
    # Create dict for quick lookup
    stats_dict = fbref_df.set_index('team').to_dict('index')
    
    # Add stats for home and away teams
    for stat_col in fbref_df.columns:
        if stat_col == 'team':
            continue
        
        matches_df[f'home_{stat_col}'] = matches_df['home_team'].map(
            lambda x: stats_dict.get(x, {}).get(stat_col, np.nan)
        )
        matches_df[f'away_{stat_col}'] = matches_df['away_team'].map(
            lambda x: stats_dict.get(x, {}).get(stat_col, np.nan)
        )
    
    return matches_df

def engineer_features(matches_df, odds_df=None, fbref_df=None):
    """Create features for each match"""
    
    # Load additional data if not provided
    if odds_df is None and os.path.exists('data/odds_history.csv'):
        odds_df = pd.read_csv('data/odds_history.csv')
    
    if fbref_df is None and os.path.exists('data/fbref_stats.csv'):
        fbref_df = pd.read_csv('data/fbref_stats.csv')
    
    # Merge additional data
    if odds_df is not None:
        matches_df = merge_odds_data(matches_df, odds_df)
    
    if fbref_df is not None:
        matches_df = merge_fbref_stats(matches_df, fbref_df)
    
    features_list = []
    
    for idx, match in matches_df.iterrows():
        historical = matches_df.iloc[:idx]
        
        if len(historical) < 10:
            continue
        
        # Calculate base features using shared function
        features = calculate_base_features(historical, match['home_team'], match['away_team'])
        features.update({
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'outcome': match['outcome'],
        })
        
        # Add betting odds features if available
        if 'home_odds' in match and pd.notna(match['home_odds']):
            features['home_odds'] = match['home_odds']
            features['draw_odds'] = match.get('draw_odds', np.nan)
            features['away_odds'] = match['away_odds']
            
            # Implied probabilities from odds
            draw_odds_val = match.get('draw_odds', 3.0)
            if pd.notna(draw_odds_val):
                total = 1/match['home_odds'] + 1/draw_odds_val + 1/match['away_odds']
                features['implied_home_prob'] = (1/match['home_odds']) / total
                features['implied_away_prob'] = (1/match['away_odds']) / total
            else:
                features['implied_home_prob'] = np.nan
                features['implied_away_prob'] = np.nan
        
        # Add FBref features if available
        fbref_features = ['xG', 'xGA', 'Poss', 'Sh', 'SoT']
        for feat in fbref_features:
            home_col = f'home_{feat}'
            away_col = f'away_{feat}'
            if home_col in match and pd.notna(match[home_col]):
                features[f'home_{feat.lower()}'] = match[home_col]
                features[f'away_{feat.lower()}'] = match[away_col]
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def train_models(features_df):
    """Train and evaluate multiple models"""
    X = features_df.drop(['outcome', 'home_team', 'away_team'], axis=1)
    y = features_df['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"\n📊 Class distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"📊 Class weights: {class_weight_dict}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models with class weights to handle imbalance
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', scale_pos_weight=None)
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
    
    # Load additional data sources
    odds_df = None
    if os.path.exists('data/odds_history.csv'):
        odds_df = pd.read_csv('data/odds_history.csv')
        print(f"✓ Loaded {len(odds_df)} odds records")
    
    fbref_df = None
    if os.path.exists('data/fbref_stats.csv'):
        fbref_df = pd.read_csv('data/fbref_stats.csv')
        print(f"✓ Loaded stats for {len(fbref_df)} teams")
    
    print(f"✓ Loaded {len(matches_df)} matches")
    
    print("\n🔧 Engineering features...")
    features_df = engineer_features(matches_df, odds_df, fbref_df)
    features_df.to_csv('data/features.csv', index=False)
    print(f"✓ Created {len(features_df)} feature records with {len(features_df.columns)} features")
    
    print("\n🤖 Training models...")
    model, scaler, features, results = train_models(features_df)
    
    return model, scaler, features, results