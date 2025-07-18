from fastapi import FastAPI
import pandas as pd
from predict import predict_new_game

batting_df = pd.read_csv("data/batting_2025.csv")
pitching_df = pd.read_csv("data/pitching_2025.csv")

app = FastAPI()

@app.get("/predict")
def predict(home_team: str, away_team: str):
    try:
        home_era = pitching_df[pitching_df['Team'] == home_team]['ERA'].values[0]
        away_era = pitching_df[pitching_df['Team'] == away_team]['ERA'].values[0]
    except IndexError:
        return {"error": "Team not found in pitching stats."}

    return predict_new_game({
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher_era": home_era,
        "away_pitcher_era": away_era
    })
