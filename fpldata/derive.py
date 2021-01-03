"""
This module contains functions for deriving new FPLManagerBase data set from existing ones.
"""
import numpy as np
import pandas as pd
from .common import Context, validate_df, value_or_default, FIXTURE_STATS_TYPES, LOCAL_COL_PREFIX
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Callable
import itertools

# Define type aliases
DF = pd.DataFrame
S = pd.Series

MIN_SHORT_POINTS = 1
MIN_LONG_POINTS = 2
DEF_GOAL_POINTS = 6
MID_GOAL_POINTS = 5
FWD_GOAL_POINTS = 4
ASSIST_POINTS = 3
PEN_MISS_POINTS = -2
GOAL_CONCEDED_POINTS = -1
DEF_CLEAN_POINTS = 4
MID_CLEAN_POINTS = 1
FWD_CLEAN_POINTS = 0
SAVES_POINTS = 1
PEN_SAVE_POINTS = 5
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2

def get_player_teams(players: DF, teams: DF, ctx: Context):
    return (players
            .reset_index(drop=False)
            .merge(teams, left_on='Player Team Code', right_on='Team Code')
            .set_index('Player Code')
            .assign(**{'Long Name': lambda df: df['First Name'] + ' ' + df['Last Name']})
            .assign(**{'Long Name and Team': lambda df: df['Long Name'] + ' (' + df['Team Name'] + ')'})
            .assign(**{'Name and Short Team': lambda df: df['Name'] + ' (' + df['Team Short Name'] + ')'})
            .pipe(ctx.dd.reorder))


def get_fixture_teams(fixtures: DF, teams: DF, ctx: Context) -> DF:
    return (fixtures
            .reset_index()
            .merge(teams[['Team Name', 'Team Short Name']]
                   .rename(columns={'Team Name': 'Team Name Home', 'Team Short Name': 'Team Short Name Home'}),
                   left_on='Home Team Code',
                   right_on='Team Code', suffixes=(False, False))
            .merge(teams[['Team Name', 'Team Short Name']]
                   .rename(columns={'Team Name': 'Team Name Away', 'Team Short Name': 'Team Short Name Away'}),
                   left_on='Away Team Code',
                   right_on='Team Code', suffixes=(False, False))
            .pipe(ctx.dd.reorder))


def get_players_history_fixtures(players_history: DF, fixtures: DF, player_teams: DF, ctx: Context) -> DF:
    return (players_history[['Fixture Total Points', 'Fixture Minutes Played', 'Fixture Cost']]
            .reset_index()
            .merge(fixtures, left_on='Fixture Code', right_index=True)
            .merge(player_teams[['Player Team Code', 'Field Position', 'Minutes Percent', 'News And Date', 'Team Short Name', 'Name and Short Team']], left_on='Player Code', right_index=True)
            .assign(**{'Fixture Played': lambda df: df['Fixture Minutes Played'] > 0})
            .assign(**{'Total Points To Fixture': lambda df: df.groupby('Player Code')['Fixture Total Points'].apply(lambda x: x.shift().rolling(ctx.player_fixtures_look_back, min_periods=1).sum()).fillna(0.0)})
            .assign(**{'Fixtures Played To Fixture': lambda df: df.groupby('Player Code')['Fixture Played'].apply(lambda x: x.shift().rolling(ctx.player_fixtures_look_back, min_periods=1).sum().fillna(method='ffill').fillna(0))})
            .set_index(['Player Code', 'Fixture Code']))


def get_total_points_estimator(player_team_stats: DF):
    """Estimate the total points for last season based on the players' cost using linear regression. It returns a function that expects a series of cost numbers and returns the estimated total points."""
    validate_df(player_team_stats, 'player_team_stats', ['Last Season Minutes', 'Last Season Total Points', 'Current Cost'])

    # Reduce the data to players that have played at least the equivalent of 16 full matches during the last season.
    min_minutes_played_last_season = 16 * 90
    player_team_stats_active = player_team_stats[lambda df: df['Last Season Minutes'] > min_minutes_played_last_season]

    # Split the data into training/testing sets
    total_points = np.reshape(player_team_stats_active['Last Season Total Points'].values, (player_team_stats_active.shape[0], 1))
    current_cost = np.reshape(player_team_stats_active['Current Cost'].values, (player_team_stats_active.shape[0], 1))  # Use current cost since start of season cost is not available
    total_points_train, total_points_test, current_cost_train, current_cost_test = train_test_split(total_points, current_cost, test_size=0.2, random_state=1)

    # Train linear regression model
    total_points_estimator = LinearRegression()
    _ = total_points_estimator.fit(current_cost_train, total_points_train)

    # Make predictions using the testing set
    # total_points_pred = total_points_estimator.predict(current_cost_test)
    # print(f'Mean squared error: {np.mean((regr.predict(current_cost_test) - total_points_test) ** 2):.2f}')

    def pred(s: pd.Series) -> pd.Series:
        index_values = s.index.values
        predictions = total_points_estimator.predict(np.reshape(s.values, (s.size, 1)))
        predictions_s = pd.Series(np.reshape(predictions, s.size), index=index_values)
        return predictions_s

    return pred


def calc_consistency(s: S):
    if s.count() == 0:
        return np.nan

    max_points = max(s)
    if max_points == 0:
        return np.nan

    return np.mean(s / max(s)) * 100


def calc_stats(df: DF, game_week: int = None) -> S:
    team_code = df['Player Team Code'].iloc[0]

    if game_week is not None:
        df = df[df['Game Week'] <= game_week]

    s = {'Total Points': df['Fixture Total Points'].sum(),
         'Total Points Consistency': calc_consistency(df['Fixture Total Points']),
         'Player Team Code': team_code}

    return S(s)


def get_last_season(players_history_past: DF) -> str:
    return (players_history_past
            .index
            .get_level_values('Season')
            .max())


def get_player_team_stats(player_teams: DF, players_history_fixtures: DF, players_history_past: DF, ctx: Context) -> DF:
    players_history_stats = (players_history_fixtures
                             .groupby(['Player Code'])
                             .apply(lambda df: calc_stats(df))
                             .pipe(ctx.dd.ensure_cols, ['Total Points', 'Total Points Consistency', 'Player Team Code']))

    player_team_stats = (player_teams
                         .reset_index()
                         .merge(players_history_stats[['Total Points Consistency']], left_on='Player Code', right_on='Player Code', how='left')
                         .set_index('Player Code')
                         .assign(**{'Points Per Cost': lambda df: df['Total Points'] / df['Current Cost']}))

    # noinspection PyTypeChecker
    last_season = players_history_past.pipe(get_last_season)

    # Add the total points from the last season to the player stats so it can be used for the expected point calculation at the beginning of the season.
    players_history_last_season = (players_history_past
                                   [['Season Total Points', 'Season Minutes', 'Season ICT Index']]
                                   .xs(last_season, axis=0, level=1, drop_level=False)
                                   .reset_index(level=1)
                                   .rename(columns={'Season Total Points': 'Last Season Total Points', 'Season Minutes': 'Last Season Minutes', 'Season ICT Index': 'Last Season ICT Index'}))

    return (player_team_stats
            .merge(players_history_last_season, left_index=True, right_index=True, how='left'))


def add_last_season_fallback(player_team_stats: DF, total_points_estimator: Callable) -> DF:
    min_minutes_played_last_season = 16 * 90
    avg_min_per_game = 80
    gws_count = 38

    return (player_team_stats
            # Take last seasons points if enough minutes played or estimate points based on player cost.
            .assign(**{'Last Season Total Points Est': lambda df: np.where(df['Last Season Minutes'] > min_minutes_played_last_season, df['Last Season Total Points'], total_points_estimator(df['Current Cost']))})
            # Estimate the game weeks played last season based on minutes played if enough minutes played or assume all game weeks played (and points estimated based on player cost).
            .assign(**{'Last Season GWs Est': lambda df: np.where(df['Last Season Minutes'] > min_minutes_played_last_season, np.minimum(df['Last Season Minutes'] / avg_min_per_game, gws_count), gws_count)}))


def phase_over(n1: float, n2: float, gw: int, speed: float = 0.2):
    """Phases over the influence of `n1` to `n2` depending on the game week `gw` at the given `speed`."""
    weight = np.tanh((gw - 1) * speed)
    return (1 - weight) * n1 + weight * n2


def add_fixtures_ago(team_fixtures: DF) -> DF:
    return (team_fixtures
            .sort_values(['Fixture Code', 'Kick Off Time'], ascending=False)
            .assign(**{'Fixtures Ago': lambda df: df.groupby(['Team Code']).cumcount() + 1}))


def get_team_fixture_scores(fixture_teams: DF, teams: DF) -> DF:
    """
    Converts the given fixture team stats (one row per fixture) to a data frame with one row for each fixture and team combination.
    The resulting data frame has twice as many rows. It then calculates stats for each row form the team's point of view and then
    adds more team information for each row.

    Args:
        fixture_teams: The fixture stats data frame.
        teams: The team data frame.

    Returns:
        A data frame with one row for each fixture and team combination.
    """
    validate_df(fixture_teams, 'fixture_teams', ['Fixture Code', 'Kick Off Time', 'Game Week', 'Home Team Score', 'Away Team Score', 'Home Team Code', 'Away Team Code'])

    # Unfold data frame so that there a two rows for each fixture.
    return (pd.melt(fixture_teams[['Fixture Code', 'Kick Off Time', 'Season', 'Game Week', 'Home Team Score', 'Away Team Score', 'Home Team Code', 'Away Team Code']],
                    id_vars=['Fixture Code', 'Kick Off Time', 'Season', 'Game Week', 'Home Team Score', 'Away Team Score'],
                    value_vars=['Home Team Code', 'Away Team Code'])
            .rename(columns={'variable': 'Variable', 'value': 'Value'})
            .sort_values(['Season', 'Game Week'])
            .assign(**{'Team Goals Scored': lambda df: np.where(df['Variable'] == 'Home Team Code', df['Home Team Score'], df['Away Team Score'])})
            .assign(**{'Team Goals Conceded': lambda df: np.where(df['Variable'] == 'Home Team Code', df['Away Team Score'], df['Home Team Score'])})
            .assign(**{'Team Clean Sheet': lambda df: np.where(df['Variable'] == 'Home Team Code', df['Away Team Score'] == 0, df['Home Team Score'] == 0)})
            .assign(**{'Is Home?': lambda df: df['Variable'] == 'Home Team Code'})
            .rename(columns={'Value': 'Team Code'}).drop('Variable', axis=1)
            .merge(teams, left_on='Team Code', right_on='Team Code'))


def calc_team_score_stats(team_fixture_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates team score stats for each fixture and team combination.

    Args:
        team_fixture_scores: The fixture and team data frame.

    Returns:
        The given data frame with the additional stats.
    """
    validate_df(team_fixture_scores, 'team_fixture_scores', ['Team Code', 'Team Short Name', 'Is Home?', 'Team Goals Scored', 'Team Goals Conceded'])

    team_score_stats = (team_fixture_scores
                        .groupby(['Team Code', 'Is Home?'])[['Team Goals Scored', 'Team Goals Conceded', 'Team Clean Sheet', 'Fixture Code']]
                        .agg({'Team Goals Scored': 'sum', 'Team Goals Conceded': 'sum', 'Team Clean Sheet': 'sum', 'Fixture Code': 'count'})
                        .rename(columns={'Fixture Code': 'Team Fixture Count'})
                        .unstack(level=-1)
                        .reset_index()
                        .rename(columns={False: 'Away', True: 'Home'}))
    team_score_stats.columns = [' '.join(col).strip() for col in team_score_stats.columns]
    team_score_stats = (team_score_stats
                        .set_index('Team Code')
                        .rename(columns={'Team Goals Conceded Home': 'Total Team Goals Conceded Home',
                                         'Team Goals Conceded Away': 'Total Team Goals Conceded Away',
                                         'Team Goals Scored Home': 'Total Team Goals Scored Home',
                                         'Team Goals Scored Away': 'Total Team Goals Scored Away',
                                         'Team Clean Sheet Home': 'Total Team Clean Sheets Home',
                                         'Team Clean Sheet Away': 'Total Team Clean Sheets Away'})
                        .assign(**{'Team Fixture Count': lambda df: df['Team Fixture Count Home'] + df['Team Fixture Count Away']})
                        .assign(**{'Total Team Goals Scored': lambda df: df['Total Team Goals Scored Away'] + df['Total Team Goals Scored Home']})
                        .assign(**{'Total Team Goals Conceded': lambda df: df['Total Team Goals Conceded Away'] + df['Total Team Goals Conceded Home']})
                        .assign(**{'Total Team Clean Sheets': lambda df: df['Total Team Clean Sheets Away'] + df['Total Team Clean Sheets Home']})
                        )

    return team_score_stats


def fill_team_goal_stats_est(team_stats: DF, team_goal_stats_est: DF, ctx: Context) -> DF:
    team_stats = (team_stats.merge(team_goal_stats_est.drop(columns=['Team Short Name']),
                                   left_index=True, right_index=True, how='outer', suffixes=(None, None))
                  .fillna(0))

    for stat_type in itertools.product(*FIXTURE_STATS_TYPES):
        post_fix = ' ' + ' '.join(stat_type).strip()
        fixture_type = (' ' + stat_type[1]).rstrip()
        team_stats = (team_stats.assign(
            **{f'_Avg Team{post_fix}':
               # Calculate average based on actual past stats if available
                   lambda df: np.where(df[f'Team Fixture Count{fixture_type}'] != 0, df[f'Total Team{post_fix}'] / df[f'Team Fixture Count{fixture_type}'], 0)
                              * np.where(df[f'Team Fixture Count{fixture_type}~'] != 0, df[f'Team Fixture Count{fixture_type}'] / df[f'Team Fixture Count{fixture_type}~'], 1)
                              # Add estimated stats if available
                              + np.where(df[f'Team Fixture Count{fixture_type}~'] != 0, (df[f'Total Team{post_fix}~'] / df[f'Team Fixture Count{fixture_type}~'])
                                         * ((df[f'Team Fixture Count{fixture_type}~'] - df[f'Team Fixture Count{fixture_type}']).clip(0, None) / df[f'Team Fixture Count{fixture_type}~']), 0)
               }))

    return (team_stats
            .assign(**{'Team Stats Quality': lambda df: df[f'Team Fixture Count'] / ctx.fixtures_look_back})
            .drop(columns=[col for col in team_stats.columns if col.endswith('~')]))  # Remove estimate columns


def fill_missing_teams(team_stats: DF, teams_ext: DF) -> DF:
    return (team_stats
            .merge(teams_ext[['Team Short Name']], left_index=True, right_index=True, how='right'))


def get_team_goal_stats(fixture_teams: DF, teams: DF, team_goal_stats_est: DF, ctx: Context) -> DF:
    return (fixture_teams
            [lambda df: df['Finished'] == True]
            .pipe(get_team_fixture_scores, teams)
            .pipe(add_fixtures_ago)
            [lambda df: df['Fixtures Ago'] <= ctx.fixtures_look_back]
            .pipe(calc_team_score_stats)
            .pipe(fill_team_goal_stats_est, team_goal_stats_est, ctx)
            .pipe(fill_missing_teams, teams))


def get_fixture_goal_stats(fixture_teams: DF, team_score_stats: DF) -> DF:
    fixture_team_stats_cols = [f'_Avg Team {" ".join(stat_type).strip()}' for stat_type in itertools.product(*FIXTURE_STATS_TYPES)] + ['Team Fixture Count Home', 'Team Fixture Count Away', 'Team Fixture Count', 'Team Stats Quality']

    return (fixture_teams
            .merge(team_score_stats[fixture_team_stats_cols].rename(columns=lambda col: col.replace('Team ', 'Home Team ')),
                   left_on='Home Team Code', right_index=True, suffixes=(None, None))
            .merge(team_score_stats[fixture_team_stats_cols].rename(columns=lambda col: col.replace('Team ', 'Away Team ')),
                   left_on='Away Team Code', right_index=True, suffixes=(None, None))
            .assign(**{'Fixture Short Name': lambda df: df['Team Short Name Home'] + '-' + df['Team Short Name Away']})
            .set_index('Fixture Code'))


def add_team_cols(team_fixture_strength: DF) -> DF:
    for col in team_fixture_strength.columns:
        if 'Away ' in col:
            away_col = col
            home_col = col.replace('Away ', 'Home ')
            col_prefix = LOCAL_COL_PREFIX if col.startswith(LOCAL_COL_PREFIX) else ''

            team_fixture_strength = (team_fixture_strength
                                     .assign(**{away_col.replace('Away ', ''): lambda df: np.where(df['Is Home?'], df[home_col], df[away_col])})
                                     .assign(**{col_prefix+'Opp '+away_col.replace('Away ', '').replace(LOCAL_COL_PREFIX, ''): lambda df: np.where(~df['Is Home?'], df[home_col], df[away_col])})
                                     .drop(columns=[away_col, home_col]))

    return team_fixture_strength


def add_fixture_stats(fixture_teams_stats: DF, ctx: Context) -> DF:
    def save_div(num: float, s: S) -> S:
        return (num / s.fillna(1)).replace(np.inf, 1)

    for stat_type in itertools.product(*FIXTURE_STATS_TYPES):
        post_fix = ' '.join(stat_type).strip()
        fixture_teams_stats = (fixture_teams_stats
                           .sort_values(['Season', 'Game Week'])
                           .assign(
                                **{f'_Avg Opp Avg Team {post_fix} To Fixture': lambda df: df.groupby('Team Code')[f'_Opp Avg Team {post_fix}']
                                    .apply(lambda x: x.shift().rolling(ctx.player_fixtures_look_back, min_periods=1).mean()).fillna(0)})
                           .assign(**{f'_Rel Opp Avg Team {post_fix} To Fixture': lambda df:
                                    np.where(df[f'_Avg Opp Avg Team {post_fix} To Fixture'] > 0, df[f'_Opp Avg Team {post_fix}'] / df[f'_Avg Opp Avg Team {post_fix} To Fixture'], 1)}))

    return (fixture_teams_stats
        .assign(**{'_Rel Att Fixture Strength Home': lambda df: df['_Rel Opp Avg Team Goals Conceded Away To Fixture'].fillna(1)})
        .assign(**{'_Rel Att Fixture Strength Away': lambda df: df['_Rel Opp Avg Team Goals Conceded Home To Fixture'].fillna(1)})
        .assign(**{'_Rel Def Fixture Strength Home': lambda df: save_div(1, df['_Rel Opp Avg Team Goals Scored Away To Fixture'])})
        .assign(**{'_Rel Def Fixture Strength Away': lambda df: save_div(1, df['_Rel Opp Avg Team Goals Scored Home To Fixture'])})
        .assign(**{'_Rel Clean Sheet Fixture Strength Home': lambda df: save_div(1, df['_Rel Opp Avg Team Clean Sheets Away To Fixture'])})
        .assign(**{'_Rel Clean Sheet Fixture Strength Away': lambda df: save_div(1, df['_Rel Opp Avg Team Clean Sheets Home To Fixture'])})
        .assign(**{'Rel Att Fixture Strength': lambda df: df['_Rel Opp Avg Team Goals Conceded To Fixture'].fillna(1)})
        .assign(**{'Rel Def Fixture Strength': lambda df: 1 / df['_Rel Opp Avg Team Goals Scored To Fixture'].fillna(1)})
        .assign(**{'Expected Goals For': lambda df: df['_Avg Opp Avg Team Goals Conceded To Fixture']})
        .assign(**{'Expected Goals Against': lambda df: df['_Avg Opp Avg Team Goals Scored To Fixture']}))


def get_team_fixture_strength(fixture_teams_stats: DF, teams: DF, ctx: Context) -> DF:
    stats_cols = [col for col in fixture_teams_stats.columns if 'Avg' in col or 'Count' in col or 'FDR' in col or 'Quality' in col]

    # noinspection PyTypeChecker
    return (pd.melt(fixture_teams_stats
                    .reset_index()
                    [['Fixture Code', 'Home Team Code', 'Team Name Home', 'Away Team Code'] + stats_cols],
                    id_vars=['Fixture Code'] + stats_cols,
                    value_vars=['Home Team Code', 'Away Team Code'])
            .drop('variable', axis=1)
            .rename(columns={'value': 'Team Code'})
            .merge(fixture_teams_stats[['Home Team Code', 'Away Team Code', 'Season', 'Game Week', 'Started', 'Fixture Short Name', 'Kick Off Time']], left_on='Fixture Code', right_index=True, suffixes=(False, False))
            .assign(**{'Is Home?': (lambda df: df['Home Team Code'] == df['Team Code'])})
            .assign(**{'Opp Team Code': lambda df: np.where(df['Is Home?'], df['Away Team Code'], df['Home Team Code'])})
            .drop(columns=['Home Team Code', 'Away Team Code'])
            .pipe(add_team_cols)
            .pipe(add_fixture_stats, ctx)
            .assign(**{'Fixture Short Name FDR': lambda df: df['Fixture Short Name'] + ' (' + df['Team FDR'].astype('str') + ')'})
            .merge(teams, left_on='Team Code', right_on='Team Code', suffixes=(None, None))
            .merge(teams[['Team Short Name']].rename(columns={'Team Short Name': 'Opp Team Short Name'}), left_on='Opp Team Code', right_on='Team Code', suffixes=(None, None))
            .pipe(add_fixtures_ago)
            .set_index(['Team Code'])
            .pipe(ctx.dd.reorder))


def get_player_team_fixture_strength(players: DF, team_fixture_strength: DF, players_history: DF, ctx: Context) -> DF:
    def roll_sum_back(df: DF, group_by: str, col: str, ctx: Context):
        return (df.groupby(group_by)
                [col]
                .shift().rolling(ctx.player_fixtures_look_back, min_periods=1).sum().fillna(0)
                .where((df['Season'] == ctx.current_season) & (df['Game Week'] <= ctx.next_gw) | (df['Season'] != ctx.current_season)))

    return (players[['Name', 'Player Team Code', 'Field Position']]
            .reset_index()
            .merge(team_fixture_strength, left_on='Player Team Code', right_index=True, suffixes=(False, False))
            .merge(players_history[['Fixture Minutes Played', 'Fixture Total Points', 'Fixture Goals Scored', 'Fixture Assists', 'Fixture Goals Conceded',
                                    'Fixture Own Goals', 'Fixture Penalties Missed', 'Fixture Yellow Cards',
                                   'Fixture Red Cards', 'Fixture Bonus Points', 'Fixture Saves', 'Fixture Clean Sheets',
                                   'Fixture Penalties Saved']],
                   left_on=['Player Code', 'Fixture Code'], right_index=True, how='left', suffixes=(False, False))
            .sort_values(['Player Code', 'Kick Off Time'])
            .set_index(['Player Code', 'Fixture Code'])
            .assign(**{'Fixture Played': lambda df: df['Fixture Minutes Played'] > 0})
            .assign(**{'Fixture Minutes >60 Played': lambda df: df['Fixture Minutes Played'] > 60})
            .assign(**{'Fixture Minutes <60 Played': lambda df: (df['Fixture Minutes Played'] < 60) & (df['Fixture Minutes Played'] > 0)})
            .assign(**{'Fixture Goals Conceded Multiples': lambda df: np.floor(df['Fixture Goals Conceded']/2)})
            .assign(**{'Fixture Saves Multiples': lambda df: np.floor(df['Fixture Saves']/3)})
            .assign(**{'Fixture Played': lambda df: df['Fixture Minutes Played'] > 0})
            .assign(**{'Total Points To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Total Points', ctx).ffill()})
            .assign(**{'Fixtures Played To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Played', ctx).ffill()})
            .assign(**{'_Minutes >60 Played To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Minutes >60 Played', ctx).ffill()})
            .assign(**{'_Minutes <60 Played To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Minutes <60 Played', ctx).ffill()})
            .assign(**{'_Minutes Played To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Minutes Played', ctx).ffill()})
            .assign(**{'_Goals Scored To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Goals Scored', ctx).ffill()})
            .assign(**{'_Assists To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Assists', ctx).ffill()})
            .assign(**{'_Own Goals To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Own Goals', ctx).ffill()})
            .assign(**{'_Penalties Missed To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Penalties Missed', ctx).ffill()})
            .assign(**{'_Yellow Cards To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Yellow Cards', ctx).ffill()})
            .assign(**{'_Red Cards To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Red Cards', ctx).ffill()})
            .assign(**{'_Bonus Points To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Bonus Points', ctx).ffill()})
            .assign(**{'_Saves Multiples To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Saves Multiples', ctx).ffill()})
            .assign(**{'_Goals Conceded Multiples To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Goals Conceded Multiples', ctx).ffill()})
            .assign(**{'_Clean Sheets To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Clean Sheets', ctx).ffill()})
            .assign(**{'_Penalties Saved To Fixture': lambda df: df.pipe(roll_sum_back, 'Player Code', 'Fixture Penalties Saved', ctx).ffill()})
            .assign(**{'_Rel Att Strength': lambda df: np.where(df['Is Home?'],
                                                                df['_Rel Att Fixture Strength Home'],
                                                                df['_Rel Att Fixture Strength Away'])})
            .assign(**{'_Rel Def Strength': lambda df: np.where(df['Is Home?'],
                                                                df['_Rel Def Fixture Strength Home'],
                                                                df['_Rel Def Fixture Strength Away'])})
            .assign(**{'_Rel Clean Sheet Strength': lambda df: np.where(df['Is Home?'],
                                                                        df['_Rel Clean Sheet Fixture Strength Home'],
                                                                        df['_Rel Clean Sheet Fixture Strength Away'])})
            .assign(**{'Stats Completeness Percent': lambda df: df['Fixtures Played To Fixture'] / ctx.player_fixtures_look_back * 100})
            .assign(**{'Rolling Avg Game Points': lambda df: df.groupby('Player Code')['Fixture Total Points'].apply(lambda x: x.rolling(ctx.player_fixtures_look_back, min_periods=1).mean())
                    .where((df['Season'] == ctx.current_season) & (df['Game Week'] < ctx.next_gw) | (df['Season'] != ctx.current_season))})
            .drop(columns=['Fixture Minutes Played', 'Fixture Total Points', 'Fixture Played']))


def calc_eps_for_next_gws(player_gw_eps: DF, ctx: Context) -> S:
    """
    Calculates the expected points for the given time horizons. NOTE that in order to improve the speed of this function which
    gets called for each player, it makes the following assumption about the player_gw_eps argument:
        1) player_gw_eps only contains data for future game weeks
        2) player_gw_eps is sorted by 'Game Week' already
        3) player_gw_eps as a column 'Expected Point With Chance Avail' which is the element-wise product of 'Expected Points'
            and 'Chance Avail Next GW'.

    Args:
        player_gw_eps: The data frame with the expected points for each player and game week combination.
        ctx: Context data, such as the next game week, the current season, the data dictionary, etc.
    Returns:
        The data frame with the expected points for the given time horizons added as columns.
    """
    # Returns an empty data frame but with the necessary columns.
    if player_gw_eps.shape[0] == 0:
        return S(list(player_gw_eps.columns.values) + ['Expected Points ' + gw for gw in list(ctx.next_gw_counts)])

    row = player_gw_eps.iloc[0]

    for gw in range(0, ctx.total_gws-ctx.next_gw+1):
        try:
            row['Expected Points GW ' + str(ctx.next_gw+gw)] = (player_gw_eps.iloc[gw]
                                                        .at['Expected Point With Chance Avail'])
        except IndexError:
            print(gw)
            print(row)

    for next_gw_post_fix, next_gw_count in ctx.next_gw_counts.items():
        future_df = player_gw_eps.iloc[:next_gw_count]
        row['Expected Points ' + next_gw_post_fix] = future_df['Expected Point With Chance Avail'].sum()

        if next_gw_count <= 8:
            row['Fixtures ' + next_gw_post_fix] = value_or_default(future_df['Fixture Short Name FDR'].str.cat(sep=', '))

    return row


def est_chance_avail(df: DF) -> S:
    """
    Estimates the chance that a player is available for the future game weeks.
    """

    chance_avail = df['Chance Avail Next GW']

    if chance_avail.shape[0] == 0:
        return chance_avail

    if 0 < chance_avail.iloc[0] <= 1:  # If the chance available is not 0 or 1 then assume that the following game week the chance is 1.
        chance_avail.iloc[1:] = 1

    return chance_avail


def get_team_future_fixtures(team_fixture_strength: DF, players_history_fixtures: DF) -> DF:
    return (team_fixture_strength[~team_fixture_strength['Fixture Code']
            .isin(players_history_fixtures.index.get_level_values(level='Fixture Code'))])[['Fixture Code', 'Game Week']]


def get_players_future_fixture_team_strengths(player_teams: DF, team_future_fixtures: DF) -> DF:
    return (player_teams
            [['Player Team Code', 'Chance Avail Next GW', 'Field Position']]
            .reset_index()
            .merge(team_future_fixtures, left_on=['Player Team Code'], right_index=True, suffixes=(False, False))
            .sort_values('Game Week')
            .set_index(['Player Code', 'Fixture Code'])
            # Project the chance available forward based on the chance available for the next game week. TODO: Need to update based on news.
            .assign(**{'Chance Avail': lambda df: df[['Chance Avail Next GW', 'Game Week']]
                    .groupby('Player Code').apply(lambda df: est_chance_avail(df).droplevel('Player Code')) if df.shape[0] > 0 else np.nan})
            .drop(columns=['Game Week']))


def get_player_fixture_stats(players_history_fixtures: DF, players_future_fixture_team_strengths: DF, player_team_fixture_strength: DF) -> DF:
    return (pd.concat([players_history_fixtures[['Fixture Total Points', 'Fixture Minutes Played', 'Fixture Cost', 'Field Position']],
                       players_future_fixture_team_strengths], sort=False)
            .drop(columns=['Player Team Code'])
            .merge(player_team_fixture_strength.drop(columns=['Field Position']), left_index=True, right_index=True, suffixes=(False, False))
            .assign(**{'Chance Avail': lambda df: df['Chance Avail'].fillna(0)}))


def calc_eps_ext(player_fixture_stats: pd.DataFrame) -> np.ndarray:
    """
    Calculates the expected points for each fixture based on past player points, position and relative fixture strength.

    Args:
        player_fixture_stats: A data frame with the pre-calculated stats indexed by player ID and fixture ID.

    Returns: Returns the expected points for each player/fixture combination as a series.

    """

    # Defines points given for each action (see https://fantasy.premierleague.com/help/rules)
    df = player_fixture_stats

    return np.where(df['Fixtures Played To Fixture'] > 0,
        (   # Attacking related points
            ((np.where(df['Field Position'].isin(['GK', 'DEF']), DEF_GOAL_POINTS, 0)
               + np.where(df['Field Position'].isin(['MID']), MID_GOAL_POINTS, 0)
               + np.where(df['Field Position'].isin(['FWD']), FWD_GOAL_POINTS, 0))
             * df['_Goals Scored To Fixture'].fillna(0)
              + ASSIST_POINTS * df['_Assists To Fixture'].fillna(0))
             * df['_Rel Att Strength'].fillna(1)

             # Defensive points
             + np.where(df['Field Position'].isin(['GK', 'DEF']), GOAL_CONCEDED_POINTS, 0)
                * df['_Goals Conceded Multiples To Fixture'].fillna(0)
             + OWN_GOAL_POINTS * df['_Own Goals To Fixture'].fillna(0)
             / df['_Rel Def Strength'].fillna(1)

             # Clean sheet points
             + (np.where(df['Field Position'].isin(['GK', 'DEF']), DEF_CLEAN_POINTS, 0)
                + np.where(df['Field Position'].isin(['MID']), MID_CLEAN_POINTS, 0)
                + np.where(df['Field Position'].isin(['FWD']), FWD_CLEAN_POINTS, 0))
             * df['_Clean Sheets To Fixture'].fillna(0)
             * df['_Rel Def Strength'].fillna(1)

             # Other points
            + PEN_MISS_POINTS * df['_Penalties Missed To Fixture'].fillna(0)
            + SAVES_POINTS * df['_Saves Multiples To Fixture'].fillna(0)
            + PEN_SAVE_POINTS * df['_Penalties Saved To Fixture'].fillna(0)
            + df['_Bonus Points To Fixture'].fillna(0)
            + YELLOW_CARD_POINTS * df['_Yellow Cards To Fixture'].fillna(0)
            + RED_CARD_POINTS * df['_Red Cards To Fixture'].fillna(0)
            + df['_Minutes >60 Played To Fixture']*MIN_LONG_POINTS+df['_Minutes <60 Played To Fixture']*MIN_SHORT_POINTS)
            / df['Fixtures Played To Fixture'],
        0)


def calc_eps_ext_simple(player_fixture_stats: pd.DataFrame) -> np.ndarray:
    """
    Calculates the expected points for each fixture based on past player points, position and relative fixture strength.

    Args:
        player_fixture_stats: A data frame with the pre-calculated stats indexed by player ID and fixture ID.

    Returns: Returns the expected points for each player/fixture combination as a series.

    """
    return np.where(player_fixture_stats['Fixtures Played To Fixture'] > 0, player_fixture_stats['Total Points To Fixture'] / player_fixture_stats['Fixtures Played To Fixture'], 0)


def get_players_fixture_team_eps(player_fixture_stats: DF) -> DF:
    return (player_fixture_stats
            .assign(**{'Expected Points': lambda df: df.pipe(calc_eps_ext)})
            .assign(**{'Expected Points Simple': lambda df: df.pipe(calc_eps_ext_simple)})
            .assign(**{'Rel Strength': lambda df: df['Expected Points']/df['Expected Points Simple']})
            )


def proj_to_gw(players_fixture_team_eps: DF) -> DF:
    def proj_to_gw_func(col: S):
        if col.name in ('Fixture Total Points', 'Fixture Minutes Played') or col.name.startswith('Expected'):
            return 'sum'

        if col.name == 'Fixture Short Name FDR':
            return ', '.join

        if 'Fixture Count' in col:
            return 'count'

        if np.issubdtype(col.dtype, np.number):
            return 'mean'

        return 'last'

    def fill_missing_gws(players_gw_team_eps: DF, player_gws: DF) -> DF:
        players_gw_team_eps = (players_gw_team_eps
                               .merge(player_gws, left_index=True, right_index=True, how='right', suffixes=(False, False)))

        # Fills some columns with 0 if there is no fixture for a player in a particular game week.
        for col in players_gw_team_eps.columns:
            if 'Expected Points' in col or 'Chance Avail' in col or 'Fixture Count' in col:
                players_gw_team_eps = (players_gw_team_eps
                                       .assign(**{col: lambda df: df[col].fillna(0.0)}))

        return players_gw_team_eps

    def nan_future_gws(players_gw_team_eps: DF) -> DF:
        # This is necessary because the results of the sum function is 0 and not np.nan for series with only pd.nan elements and calling sum with min_count=1 is too slow.
        return (players_gw_team_eps
                .assign(**{'Fixture Total Points': lambda df: np.where(df['Fixture Cost'].isnull(), np.nan, df['Fixture Total Points'])})
                .assign(**{'Fixture Minutes Played': lambda df: np.where(df['Fixture Cost'].isnull(), np.nan, df['Fixture Minutes Played'])}))

    def get_player_gws(players_gw_team_eps: DF) -> pd.Index:
        # Create a data frame with a row of every game week/player ID combination for the current and the last season. This is required to deal with game weeks that have double or missing fixtures.
        return (players_gw_team_eps
                [['Season', 'Game Week']]
                .groupby(['Season', 'Game Week'])
                .size()
                .reset_index()
                .drop(columns=[0])
                .assign(**{'Key': 1})
                .merge(players_gw_team_eps.index.unique(level=0).to_frame().assign(**{'Key': 1}), on='Key')
                .drop(columns='Key')).set_index(['Player Code', 'Season', 'Game Week'])

    # noinspection PyTypeChecker
    player_gws = players_fixture_team_eps.pipe(get_player_gws)

    # Projects from fixtures to game weeks.
    players_fixture_team_eps = players_fixture_team_eps.assign(**{'Fixture Count': 1})

    return (players_fixture_team_eps
            .groupby(['Player Code', 'Season', 'Game Week'])
            .agg({col: proj_to_gw_func(players_fixture_team_eps[col]) for col in players_fixture_team_eps.columns})
            .drop(columns=['Game Week', 'Season'])
            .pipe(fill_missing_gws, player_gws)
            # TODO: Fill forward all to fixture columns .assign(**{'Fixtures Played To Fixture': lambda df: df.groupby('Player Code')['Fixtures Played To Fixture'].transform(lambda v: v.ffill())})
            .pipe(nan_future_gws)
            )


def get_players_gw_team_eps(players_fixture_team_eps: DF, player_teams: DF) -> DF:
    """
    The resulting data frame contains one row for every player and game week in the current season.
    """
    # noinspection PyTypeChecker
    return (players_fixture_team_eps
            .drop(columns=['Field Position', 'Name', 'Team ID', 'Team Short Name', 'Team Name', 'Team Strength', 'Player Team Code', 'Chance Avail Next GW', 'Team Last Updated'])
            .pipe(proj_to_gw)
            .merge(player_teams.drop(columns=['Season']), left_on='Player Code', right_index=True, suffixes=(None, None)))


def get_player_gw_next_eps(players_gw_team_eps: DF, ctx: Context) -> DF:
    return (players_gw_team_eps
            .reset_index()
            [lambda df: (df['Game Week'] >= ctx.next_gw) & (df['Season'] == ctx.current_season)]
            .sort_values(['Season', 'Game Week'])
            .assign(**{'Expected Point With Chance Avail': lambda df: df['Expected Points'] * df['Chance Avail Next GW'] / 100})
            .groupby('Player Code')
            .apply(lambda df: df.pipe(calc_eps_for_next_gws, ctx))
            .drop(columns=['Player Code']))
