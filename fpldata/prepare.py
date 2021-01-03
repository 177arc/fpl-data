"""
This module contains functions for preparing data that was extracted from the FPLManagerBase API for the calculations to follow.
"""

import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict
from .common import Context, POSITION_BY_TYPE, STATS_TYPES
import collections

# Define type aliases
DF = pd.DataFrame
S = pd.Series


def get_next_gw_name(next_gw: int) -> str:
    if next_gw == 1:
        return 'Next GW'

    return f'Next {next_gw} GWs'


def get_next_gw_counts(ctx: Context) -> Dict[str, int]:
    return collections.OrderedDict([(get_next_gw_name(gw), gw) for gw in range(1, ctx.total_gws - ctx.next_gw + 1)])


def get_news(row: S):
    """Derives the text for the News column."""
    if pd.isnull(row['News']) or row['News'] == '':
        return None

    date_part = '' if pd.isnull(row['News Date'] or row['News Date'] == 'None') else ' (' + dt.datetime.strftime(row['News Date'], '%d %b %Y') + ')'
    return str(row['News']) + date_part


def prepare_players(players_raw: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    return (players_raw
            .pipe(ctx.dd.remap, data_set='player')
            .pipe(ctx.dd.strip_cols, data_set='player')
            .assign(**{'ICT Index': lambda df: pd.to_numeric(df['ICT Index'])})
            .assign(**{'Field Position': lambda df: df['Field Position Code'].map(lambda x: POSITION_BY_TYPE[x])})
            .assign(**{'Current Cost': lambda df: df['Current Cost x10'] / 10})
            .assign(**{'Minutes Percent': lambda df: df['Minutes Played'] / df['Minutes Played'].max() * 100 if ctx.next_gw > 1 else 0})
            .assign(**{'News And Date': lambda df: df.apply(lambda row: get_news(row), axis=1)})
            .assign(**{'Percent Selected': lambda df: pd.to_numeric(df['Percent Selected'])})
            .assign(**{'Chance Avail This GW': lambda df: df['Chance Avail This GW'].map(lambda x: x if not pd.isnull(x) else 100)})
            .assign(**{'Chance Avail Next GW': lambda df: df['Chance Avail Next GW'].map(lambda x: x if not pd.isnull(x) else 100)})
            .assign(**{'Player Last Updated': ctx.now})
            .rename_axis('Player ID')
            .reset_index()
            .set_index('Player Code')
            .pipe(ctx.dd.reorder))


def prepare_players_history_past(players_history_past_raw: DF, players_id_code_map: S, ctx: Context) -> DF:
    return (players_history_past_raw
            .pipe(ctx.dd.remap, data_set='players_history_past')
            .pipe(ctx.dd.strip_cols, data_set='players_history_past')
            .rename_axis(['Player ID', 'Season'])
            .reset_index()
            .merge(players_id_code_map, left_on='Player ID', right_index=True)
            .set_index(['Player Code', 'Season']))


def prepare_players_history(players_history_raw: DF, players_id_code_map: S, fixtures_id_code_map: S, ctx: Context) -> DF:
    return (players_history_raw
            .pipe(ctx.dd.remap, data_set='player_hist')
            .pipe(ctx.dd.strip_cols, data_set='player_hist')
            .rename_axis(['Player ID', 'Fixture ID'])
            .reset_index()
            .merge(players_id_code_map, left_on='Player ID', right_index=True)
            .merge(fixtures_id_code_map, left_on='Fixture ID', right_index=True)
            .set_index(['Player Code', 'Fixture Code'])
            .pipe(ctx.dd.ensure_cols, data_set='player_hist')
            .assign(**{'Fixture Cost': lambda df: df['Fixture Cost x10'] / 10})
            .assign(**{'Fixture ICT Index': lambda df: pd.to_numeric(df['Fixture ICT Index'])}))


def prepare_teams(teams_raw: DF, ctx: Context) -> DF:
    return (teams_raw
            .pipe(ctx.dd.remap, 'team')
            .pipe(ctx.dd.strip_cols, 'team')
            .assign(**{'Team Last Updated': ctx.now})
            .rename_axis('Team ID')
            .reset_index()
            .set_index('Team Code'))


def prepare_fixtures(fixtures_raw: DF, team_id_code_map: S, ctx: Context) -> DF:
    return (fixtures_raw
            .pipe(ctx.dd.remap, 'fixture')
            .pipe(ctx.dd.strip_cols, 'fixture')
            .rename_axis('Fixture ID')
            .reset_index()
            .set_index('Fixture Code')
            .merge(team_id_code_map.rename('Away Team Code'), left_on='Away Team ID', right_index=True)
            .merge(team_id_code_map.rename('Home Team Code'), left_on='Home Team ID', right_index=True)
            .sort_values('Game Week')
            # If game week is still unknown, assume that the fixture will happen in the last game week so that expected points calculation to the end of the season is correct.
            .assign(**{'Game Week': lambda df: np.where(~df['Game Week'].isnull(), df['Game Week'], ctx.total_gws).astype('int64')})
            .assign(**{'Fixture Last Updated': ctx.now})
            .pipe(ctx.dd.reorder))


def prepare_game_weeks(game_week_raw: DF, ctx: Context) -> DF:
    return (game_week_raw
            .pipe(ctx.dd.remap, 'game_week')
            .pipe(ctx.dd.strip_cols, 'game_week')
            .rename_axis('GW ID')
            .assign(**{'GW Last Updated': ctx.now}))


def cache_hash(args: list, kargs: dict) -> int:
    """
    Ensures that the hash calc does not trigger pickling which fails for FPLPandas objects.
    """
    return hash((args, frozenset(kargs.items())))


def get_team_id_code_map(teams: DF) -> S:
    return teams[['Team ID']].reset_index().set_index('Team ID')['Team Code']


def get_fixtures_id_code_map(fixtures: DF) -> S:
    return fixtures[['Fixture ID']].reset_index().set_index('Fixture ID')['Fixture Code']


def get_players_id_code_map(players: DF) -> S:
    return players[['Player ID']].reset_index().set_index('Player ID')['Player Code']


def get_team_goal_stats_est(last_season_stats_est: DF) -> DF:
    for stats_type in STATS_TYPES:
        last_season_stats_est = last_season_stats_est.assign(
            **{f'Total Team {stats_type}~': lambda df: df[f'Total Team {stats_type} Home~'] + df[f'Total Team {stats_type} Away~']})

    return last_season_stats_est.assign(
        **{f'Team Fixture Count~': lambda df: df[f'Team Fixture Count Home~'] + df[f'Team Fixture Count Away~']})
