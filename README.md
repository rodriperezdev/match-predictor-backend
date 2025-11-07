# Football Match Predictor

Machine learning-based football match prediction system for the Argentine Primera División. This project uses Python, FastAPI, scikit-learn, and XGBoost to predict match outcomes based on team form, statistics, and historical data.

## Features

- **Data Collection**: Automated collection of match data from API-Football
- **Betting Odds Integration**: Collects odds from The Odds API for enhanced predictions
- **Advanced Statistics**: Scrapes xG, possession, shots, and other advanced stats from FBref
- **Feature Engineering**: Calculates team form, goal differences, head-to-head records, betting odds, and advanced stats
- **Machine Learning**: Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) and selects the best performer
- **RESTful API**: FastAPI-based API for making predictions
- **Model Accuracy**: Achieves approximately 55-65% accuracy on test data with odds and advanced stats (vs 40-50% with basic features)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rodri-perezz1998/match-predictor.git
   cd match-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv PredictorEnv
   PredictorEnv\Scripts\activate  # Windows
   # or
   source PredictorEnv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory (copy from `.env.example`):
   ```
   FOOTBALL_API_KEY=your_api_football_key_here
   ODDS_API_KEY=your_odds_api_key_here
   ```
   
   - Get your API-Football key from: https://www.api-football.com/
   - Get your Odds API key from: https://the-odds-api.com/ (500 requests/month free)

5. **Collect data** (optional, if you have API access)
   ```bash
   python run_collection.py
   ```

6. **Train the model**
   ```bash
   python run_training.py
   ```

7. **Run the API server**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8003`

## API Endpoints

- `GET /` - API status
- `GET /teams` - Get list of all teams
- `GET /stats` - Get dataset statistics
- `GET /recent-matches/{team_name}?limit=5` - Get recent matches for a team
- `POST /predict` - Make a prediction
  ```json
  {
    "home_team": "Boca Juniors",
    "away_team": "River Plate"
  }
  ```

## Project Structure

```
match-predictor/
├── api.py                 # FastAPI application
├── data_collection.py     # Data collection from API-Football
├── model_training.py      # Model training and feature engineering
├── main.py               # API server entry point
├── run_collection.py     # Script to collect data
├── run_training.py       # Script to train models
├── requirements.txt      # Python dependencies
├── data/                 # Match data (CSV files)
└── models/               # Trained models (pickle files)
```

## Technologies

- **Python 3.x**
- **FastAPI** - Web framework
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **pandas** - Data manipulation
- **requests** - HTTP requests
- **python-dotenv** - Environment variables

## License

MIT

