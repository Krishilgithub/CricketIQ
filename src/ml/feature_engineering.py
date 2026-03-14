import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import BRONZE_CRICSHEET_DIR

def load_and_filter_data(data_dir=str(BRONZE_CRICSHEET_DIR)):
    base_dir = Path(data_dir)
    
    matches = pd.read_csv(base_dir / 'matches.csv', low_memory=False)
    deliveries = pd.read_csv(base_dir / 'deliveries.csv', low_memory=False)
    
    # 1. Enforce match filter & male only
    matches = matches[
        (matches['gender'] == 'male') & 
        (matches['match_type'] == 'T20') & 
        (matches['overs'] == 20)
    ].copy()
    
    matches['winner'] = matches['winner'].fillna('no_result')
    matches['city'] = matches['city'].fillna('Unknown')
    
    matches['match_date'] = pd.to_datetime(matches['match_date'], errors='coerce')
    matches = matches.sort_values('match_date').reset_index(drop=True)
    
    valid_ids = set(matches['match_id'])
    deliveries = deliveries[deliveries['match_id'].isin(valid_ids)].copy()
    
    match_teams = pd.read_csv(base_dir / 'match_teams.csv', low_memory=False)
    match_teams = match_teams[match_teams['match_id'].isin(valid_ids)]
    
    teams_pivot = match_teams.groupby('match_id')['team'].apply(list).reset_index()
    teams_pivot['team1'] = teams_pivot['team'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    teams_pivot['team2'] = teams_pivot['team'].apply(lambda x: x[1] if len(x) > 1 else 'Unknown')
    
    matches = matches.merge(teams_pivot[['match_id', 'team1', 'team2']], on='match_id', how='left')
    return matches, deliveries

def get_rolling_team_stats(matches, deliveries):
    deliveries['is_boundary'] = deliveries['runs_batter'].isin([4, 6])
    deliveries['is_dot'] = deliveries['runs_batter'] == 0
    deliveries['phase'] = pd.cut(
        deliveries['over'],
        bins=[-1, 5, 14, 20],
        labels=['PP', 'Middle', 'Death']
    )
    
    inning_summary = deliveries.groupby(['match_id', 'batting_team']).agg(
        total_runs=('runs_total', 'sum'),
        total_balls=('ball_in_over', 'count'),
        boundaries=('is_boundary', 'sum'),
        dots=('is_dot', 'sum')
    ).reset_index()
    
    phase_summary = deliveries.groupby(['match_id', 'batting_team', 'phase'], observed=False).agg(
        phase_runs=('runs_total', 'sum'),
        phase_balls=('ball_in_over', 'count')
    ).reset_index()
    
    phase_pivot = phase_summary.pivot_table(
        index=['match_id', 'batting_team'], 
        columns='phase', 
        values=['phase_runs', 'phase_balls'],
        fill_value=0
    ).reset_index()
    
    phase_pivot.columns = ['match_id', 'batting_team'] + [
        f'{c[1]}_{c[0]}' for c in phase_pivot.columns[2:]
    ]
    
    innings = inning_summary.merge(phase_pivot, on=['match_id', 'batting_team'], how='left')
    
    team_history = []
    for team in matches['team1'].unique().tolist() + matches['team2'].unique().tolist():
        team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].copy()
        
        for idx, row in team_matches.iterrows():
            m_id = row['match_id']
            m_date = row['match_date']
            
            past_m = team_matches[team_matches['match_date'] < m_date].tail(10)
            
            if len(past_m) == 0:
                stats = {
                    'match_id': m_id,
                    'team': team,
                    'recent_win_rate': 0.5,
                    'avg_score': 140.0,
                    'avg_rr': 7.0,
                    'avg_pp_rr': 7.5,
                    'avg_death_rr': 9.0,
                    'avg_boundaries': 15.0
                }
            else:
                wins = (past_m['winner'] == team).sum()
                win_rate = wins / len(past_m)
                
                past_innings = innings[(innings['batting_team'] == team) & (innings['match_id'].isin(past_m['match_id']))]
                if len(past_innings) > 0:
                    avg_score = past_innings['total_runs'].mean()
                    total_balls = past_innings['total_balls'].sum()
                    avg_rr = past_innings['total_runs'].sum() / (total_balls / 6) if total_balls > 0 else 7.0
                    
                    pp_balls = past_innings['PP_phase_balls'].sum()
                    avg_pp_rr = past_innings['PP_phase_runs'].sum() / (pp_balls / 6) if pp_balls > 0 else 7.5
                    
                    death_balls = past_innings['Death_phase_balls'].sum()
                    avg_death_rr = past_innings['Death_phase_runs'].sum() / (death_balls / 6) if death_balls > 0 else 9.0
                    
                    avg_boundaries = past_innings['boundaries'].mean()
                else:
                    avg_score = 140.0
                    avg_rr = 7.0
                    avg_pp_rr = 7.5
                    avg_death_rr = 9.0
                    avg_boundaries = 15.0
                    
                stats = {
                    'match_id': m_id,
                    'team': team,
                    'recent_win_rate': win_rate,
                    'avg_score': avg_score,
                    'avg_rr': avg_rr,
                    'avg_pp_rr': avg_pp_rr,
                    'avg_death_rr': avg_death_rr,
                    'avg_boundaries': avg_boundaries
                }
            team_history.append(stats)
            
    return pd.DataFrame(team_history)

def build_match_features(conn=None, source: str = 't20i') -> pd.DataFrame:
    matches, deliveries = load_and_filter_data()
    if matches.empty:
        return pd.DataFrame()
        
    venue_stats = matches.groupby('venue').agg(matches_hosted=('match_id', 'nunique')).reset_index()
    first_inn = deliveries[deliveries['innings_number'] == 1].groupby('match_id')['runs_total'].sum().reset_index()
    first_inn = first_inn.merge(matches[['match_id', 'venue']], on='match_id')
    v_1st = first_inn.groupby('venue')['runs_total'].mean().reset_index().rename(columns={'runs_total': 'venue_avg_1st_score'})
    
    second_inn = deliveries[deliveries['innings_number'] == 2].groupby('match_id')['runs_total'].sum().reset_index()
    second_inn = second_inn.merge(matches[['match_id', 'venue']], on='match_id')
    v_2nd = second_inn.groupby('venue')['runs_total'].mean().reset_index().rename(columns={'runs_total': 'venue_avg_2nd_score'})
    
    venue_stats = venue_stats.merge(v_1st, on='venue', how='left').merge(v_2nd, on='venue', how='left')
    venue_stats.fillna({'venue_avg_1st_score': 150, 'venue_avg_2nd_score': 140}, inplace=True)
    
    team_stats_df = get_rolling_team_stats(matches, deliveries)
    features = []
    for idx, match in matches.iterrows():
        m_id = match['match_id']
        team1 = match['team1']
        team2 = match['team2']
        
        target = 1 if match['winner'] == team1 else (0 if match['winner'] == team2 else None)
        row = {'match_id': m_id, 'match_date': match['match_date'], 'team1': team1, 'team2': team2, 'target': target}
        
        row['toss_winner_is_team1'] = 1 if match['toss_winner'] == team1 else 0
        row['toss_elected_bat'] = 1 if match['toss_decision'] == 'bat' else 0
        row['is_world_cup'] = 1 if 'World Cup' in str(match.get('event_name', '')) else 0
        
        t1_stats = team_stats_df[(team_stats_df['match_id'] == m_id) & (team_stats_df['team'] == team1)]
        t2_stats = team_stats_df[(team_stats_df['match_id'] == m_id) & (team_stats_df['team'] == team2)]
        
        for pfix, stats_part in [('team1', t1_stats), ('team2', t2_stats)]:
            if not stats_part.empty:
                s = stats_part.iloc[0]
                row[f'{pfix}_recent_win_rate'] = s['recent_win_rate']
                row[f'{pfix}_avg_score'] = s['avg_score']
                row[f'{pfix}_avg_rr'] = s['avg_rr']
                row[f'{pfix}_avg_pp_rr'] = s['avg_pp_rr']
                row[f'{pfix}_avg_death_rr'] = s['avg_death_rr']
                row[f'{pfix}_avg_boundaries'] = s['avg_boundaries']
            else:
                row[f'{pfix}_recent_win_rate'] = 0.5
                row[f'{pfix}_avg_score'] = 140
                row[f'{pfix}_avg_rr'] = 7.0
                row[f'{pfix}_avg_pp_rr'] = 7.5
                row[f'{pfix}_avg_death_rr'] = 9.0
                row[f'{pfix}_avg_boundaries'] = 15

        h2h = matches[(((matches['team1'] == team1) & (matches['team2'] == team2)) | ((matches['team1'] == team2) & (matches['team2'] == team1))) & (matches['match_date'] < match['match_date'])]
        if not h2h.empty:
            row['h2h_team1_win_rate'] = (h2h['winner'] == team1).sum() / len(h2h)
            row['h2h_total_matches'] = len(h2h)
        else:
            row['h2h_team1_win_rate'] = 0.5
            row['h2h_total_matches'] = 0
            
        v = venue_stats[venue_stats['venue'] == match['venue']]
        if not v.empty:
            v = v.iloc[0]
            row['venue_avg_1st_score'] = v['venue_avg_1st_score']
            row['venue_avg_2nd_score'] = v['venue_avg_2nd_score']
            row['venue_matches'] = v['matches_hosted']
        else:
            row['venue_avg_1st_score'] = 150
            row['venue_avg_2nd_score'] = 140
            row['venue_matches'] = 0
            
        row['win_rate_diff'] = row['team1_recent_win_rate'] - row['team2_recent_win_rate']
        row['avg_score_diff'] = row['team1_avg_score'] - row['team2_avg_score']
        row['avg_rr_diff'] = row['team1_avg_rr'] - row['team2_avg_rr']
        features.append(row)
        
    feat_df = pd.DataFrame(features)
    feat_df.dropna(subset=['target'], inplace=True)
    return feat_df

def build_player_features(conn=None, source: str = 't20i') -> pd.DataFrame:
    return pd.DataFrame()

def build_innings_features(conn=None, source: str = 't20i') -> pd.DataFrame:
    return pd.DataFrame()

def build_live_state_features(conn=None) -> pd.DataFrame:
    matches, deliveries = load_and_filter_data()
    deliveries = deliveries.merge(matches[['match_id', 'winner', 'team1', 'team2']], on='match_id', how='left')
    return deliveries

if __name__ == '__main__':
    print("Loading data and strictly ensuring we have men's T20 data...")
    matches, deliveries = load_and_filter_data()
    print(f"Loaded {len(matches)} matches and {len(deliveries)} deliveries after filtering for men's T20.")
    print("Unique genders after filter:", matches['gender'].unique())
    print("\nBuilding match features...")
    match_features = build_match_features()
    print(f"Generated {len(match_features)} match-level feature records.")
    
    # Save the cleaned features into models or a processed data directory
    output_path = Path(__file__).parent.parent.parent / "data" / "silver" / "processed_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_csv(output_path, index=False)
    print(f"Saved processed features to {output_path}")
