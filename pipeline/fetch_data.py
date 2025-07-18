from pybaseball import team_batting, team_pitching, schedule_and_record
import pandas as pd
import os

def fetch_team_stats(year: int):
    try:
        batting = team_batting(year)
        pitching = team_pitching(year)
        return batting, pitching
    except Exception as e:
        print(f"❌ Could not fetch data for year {year}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def fetch_game_schedule(year: int, team_abbr: str):
    return schedule_and_record(year, team_abbr)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
