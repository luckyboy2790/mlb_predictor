import pandas as pd
import numpy as np
from pybaseball import schedule_and_record

def compute_rolling_stats(df: pd.DataFrame, value_col: str, window: int = 5) -> pd.Series:
    return df.groupby('Team')[value_col].transform(lambda x: x.rolling(window, min_periods=1).mean())

def compute_home_away_split(df: pd.DataFrame) -> pd.DataFrame:
    df['is_home'] = df['Home_Away'] == 'Home'
    grouped = df.groupby(['Team', 'is_home'])

    agg = grouped.agg({
        'OPS': 'mean',
        'ERA': 'mean'
    }).reset_index()

    home_df = agg[agg['is_home']].rename(columns={
        'OPS': 'home_ops_avg',
        'ERA': 'home_era_avg'
    })[['Team', 'home_ops_avg', 'home_era_avg']]

    away_df = agg[~agg['is_home']].rename(columns={
        'OPS': 'away_ops_avg',
        'ERA': 'away_era_avg'
    })[['Team', 'away_ops_avg', 'away_era_avg']]

    return pd.merge(home_df, away_df, on='Team', how='outer')

def compute_head_to_head_stats(year: int, team_abbr: str, opponent_abbr: str) -> dict:
    try:
        sched = schedule_and_record(year, team_abbr)
    except Exception:
        return {'head_to_head_win_pct': 0.5, 'head_to_head_games': 0}

    mask = sched['Opp'] == opponent_abbr
    games = sched[mask]
    if games.empty:
        return {'head_to_head_win_pct': 0.5, 'head_to_head_games': 0}

    wins = games['W/L'].str.startswith('W').sum()
    win_pct = wins / len(games)
    return {
        'head_to_head_win_pct': win_pct,
        'head_to_head_games': len(games)
    }

def get_recent_win_pct(df: pd.DataFrame, team: str, last_n: int = 5) -> float:
    df = df[df['Tm'] == team]
    
    date_format = "%A, %b %d"
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format=date_format)
    
    df = df.sort_values('Date', ascending=False)
    
    recent_games = df.head(last_n)

    if 'W/L' not in recent_games.columns:
        print(f"⚠️ Missing W/L column for {team}")
        return 0.5
    
    wins = recent_games['W/L'].str.startswith('W').sum()
    
    if len(recent_games) > 0:
        win_pct = round(wins / len(recent_games), 3)
    else:
        win_pct = 0.5

    return win_pct

def extract_pitcher_stats(pitcher_df: pd.DataFrame, name: str) -> dict:
    try:
        row = pitcher_df[pitcher_df['Name'].str.contains(name, case=False)].iloc[0]
        return {
            'pitcher_era': row['ERA'],
            'pitcher_whip': row.get('WHIP', np.nan),
            'pitcher_k9': row.get('SO9', np.nan)
        }
    except IndexError:
        return {
            'pitcher_era': np.nan,
            'pitcher_whip': np.nan,
            'pitcher_k9': np.nan
        }

def build_feature_vector(df, team_schedules, home_team, away_team, year=2025):
    home_row = df[df['home_team'] == home_team]
    away_row = df[df['away_team'] == away_team]

    if home_row.empty or away_row.empty:
        raise ValueError("Missing team data for input prediction")

    home_ops = home_row['home_ops'].values[0]
    away_ops = away_row['away_ops'].values[0]
    home_era = home_row['home_pitcher_era'].values[0]
    away_era = away_row['away_pitcher_era'].values[0]

    h2h = compute_head_to_head_stats(year, home_team, away_team)

    try:
        home_sched = team_schedules[home_team]
        away_sched = team_schedules[away_team]
        home_win_pct = get_recent_win_pct(home_sched, home_team)
        away_win_pct = get_recent_win_pct(away_sched, away_team)
    except:
        home_win_pct = 0.5
        away_win_pct = 0.5

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher_era": home_era,
        "away_pitcher_era": away_era,
        "home_ops": home_ops,
        "away_ops": away_ops,
        "head_to_head_win_pct": h2h['head_to_head_win_pct'],
        "head_to_head_games": h2h['head_to_head_games'],
        "home_win_pct_last5": home_win_pct,
        "away_win_pct_last5": away_win_pct
    }

