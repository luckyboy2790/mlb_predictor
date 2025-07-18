import os
import pandas as pd
from pipeline.fetch_data import fetch_team_stats, save_csv
from pipeline.preprocess import preprocess_data
from pipeline.train_model import train_model
from predict import predict_new_game
from pipeline.features import (
    compute_rolling_stats,
    compute_head_to_head_stats,
    get_recent_win_pct,
    build_feature_vector
)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pybaseball import schedule_and_record

def main():
    print("🔄 Fetching updated team stats for 2025...")
    batting, pitching = fetch_team_stats(2025)

    if batting.empty or pitching.empty:
        print("⛔ No team stats found — skipping training.")
        return

    save_csv(batting, 'data/batting_2025.csv')
    save_csv(pitching, 'data/pitching_2025.csv')

    print("🧪 Checking for or creating processed dataset...")
    processed_path = 'data/processed_2025.csv'

    print("📊 Generating processed training data from stats...")
    batting = batting[['Team', 'OPS']]
    pitching = pitching[['Team', 'ERA', 'WHIP']].copy()

    if not os.path.exists(processed_path):
        batting['game_num'] = batting.groupby('Team').cumcount() + 1
        pitching['game_num'] = pitching.groupby('Team').cumcount() + 1

        batting['OPS_rolling'] = compute_rolling_stats(batting, 'OPS', window=5)
        pitching['ERA_rolling'] = compute_rolling_stats(pitching, 'ERA', window=5)

        df = pd.merge(
            batting[['Team', 'game_num', 'OPS_rolling']],
            pitching[['Team', 'game_num', 'ERA_rolling']],
            on=['Team', 'game_num']
        )

        df.columns = ['team', 'game_num', 'ops', 'era']

        teams = df['team'].unique()
        
        print("Teams:", len(teams))
        
        print("📥 Caching schedules for all teams...")
        team_schedules = {}

        number = 0
        for team in teams:
            number += 1
            print(number)
            try:
                team_schedules[team] = schedule_and_record(2025, team)
            except Exception as e:
                print(f"⚠️ Could not fetch schedule for {team}: {e}")
                team_schedules[team] = pd.DataFrame()
        
        matchups = [(home, away) for home in teams for away in teams if home != away]

        print("Number of matchups:", len(matchups))

        data = []
        print("Generating enriched matchups...")
        for home, away in matchups:
            number += 1
            print(number)
            
            home_row = df[df['team'] == home]
            away_row = df[df['team'] == away]

            if home_row.empty or away_row.empty:
                continue

            home_ops = home_row['ops'].values[0]
            away_ops = away_row['ops'].values[0]
            home_era = home_row['era'].values[0]
            away_era = away_row['era'].values[0]

            h2h_stats = compute_head_to_head_stats(2025, home, away)

            try:
                home_sched = team_schedules[home]
                away_sched = team_schedules[away]
                home_win_pct = get_recent_win_pct(home_sched, home)
                away_win_pct = get_recent_win_pct(away_sched, away)
            except Exception:
                home_win_pct = 0.5
                away_win_pct = 0.5

            # Pitcher stats
            # home_pitcher_stats = extract_pitcher_stats(pitching, home)
            # away_pitcher_stats = extract_pitcher_stats(pitching, away)

            data.append({
                'home_team': home,
                'away_team': away,
                'home_pitcher_era': home_era,
                'away_pitcher_era': away_era,
                'home_ops': home_ops,
                'away_ops': away_ops,
                'head_to_head_win_pct': h2h_stats['head_to_head_win_pct'],
                'head_to_head_games': h2h_stats['head_to_head_games'],
                'home_win_pct_last5': home_win_pct,
                'away_win_pct_last5': away_win_pct,
                # 'home_pitcher_whip': home_pitcher_stats['pitcher_whip'],
                # 'away_pitcher_whip': away_pitcher_stats['pitcher_whip'],
                # 'home_pitcher_k9': home_pitcher_stats['pitcher_k9'],
                # 'away_pitcher_k9': away_pitcher_stats['pitcher_k9'],
                'result': int(home_ops > away_ops)
            })

        pd.DataFrame(data).to_csv(processed_path, index=False)
        print("✅ Processed dataset saved.")
    else: 
        print("✅ Processed dataset already exists. Skipping fetch/generate.")
        df = pd.DataFrame()  # placeholder
        try:
            df = pd.read_csv(processed_path)
            df = df.rename(columns={
                'home_team': 'team',
                'home_ops': 'ops',
                'home_pitcher_era': 'era'
            })[['team', 'ops', 'era']]
        except Exception as e:
            print("⚠️ Failed to load df from processed CSV:", e)

        team_schedules = {}
        schedule_dir = "data/schedules"
        teams = df['team'].unique() if not df.empty else []
        for team in teams:
            sched_path = f"{schedule_dir}/{team}.csv"
            if os.path.exists(sched_path):
                team_schedules[team] = pd.read_csv(sched_path)
            else:
                team_schedules[team] = pd.DataFrame()

    print("⚙️ Preprocessing data...")
    preprocess_data(processed_path, 'model/')

    print("🤖 Training model...")
    train_model('model/X.csv', 'model/y.csv', 'model/')

    print("✅ Retraining complete.")

    print("🔮 Running test prediction...")
    home_team = "HOU"
    away_team = "TEX"

    try:
        input_game = build_feature_vector(df, team_schedules, home_team, away_team)
        print(input_game)
        result = predict_new_game(input_game)
        print(result)
    except Exception as e:
        print(f"❌ Could not predict game: {e}")

if __name__ == '__main__':
    main()
