import pandas as pd
import datetime as dt
from .manager import FPLManagerBase, Context
from typing import Tuple, NoReturn, Dict
from pandas.testing import assert_frame_equal

# Define type aliases
DF = pd.DataFrame


class FPLManagerTest(FPLManagerBase):
    tests_path: str

    def __init__(self, tests_path):
        self.tests_path = tests_path
        self.mode = 'Test'
        super().__init__()

    def create_context(self) -> Context:
        ctx = Context()
        ctx.fixtures_look_back = 20
        ctx.player_fixtures_look_back = 12
        ctx.last_season = '2019-20'
        ctx.current_season = '2020-21'
        ctx.now = dt.datetime(2020, 7, 23)
        return ctx

    def get_game_weeks(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/gws_test.csv', index_col=['id'], na_values='None')

    def get_teams(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/teams_test.csv', index_col=['id'], na_values='None')

    def get_teams_last_season(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/teams_last_season_test.csv', index_col=['id'], na_values='None')

    def get_fixtures(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/fixtures_test.csv', index_col=['id'], na_values='None')

    def get_fixtures_last_season(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/fixtures_last_season_test.csv', index_col=['id'], na_values='None')

    def get_players(self) -> Tuple[DF, DF, DF]:
        return (pd.read_csv(f'{self.tests_path}/players_test.csv', index_col=['id'], na_values='None'),
                None,
                pd.read_csv(f'{self.tests_path}/players_history_test.csv', index_col=['player_id', 'fixture'], na_values='None'))

    def get_players_last_season(self) -> Tuple[DF, DF, DF]:
        return (pd.read_csv(f'{self.tests_path}/players_last_season_test.csv', index_col=['id'], na_values='None'),
                None,
                pd.read_csv(f'{self.tests_path}/players_history_last_season_test.csv', index_col=['player_id', 'fixture'], na_values='None'))

    def get_last_season_stats_est(self) -> DF:
        # Loads estimates for team for which no history is available, in particular for those that have been promoted to the Premier League.
        return pd.read_csv(f'{self.tests_path}/team_goals_stats_estimates_test.csv').set_index('Team Code')

    def publish_data_sets(self, variables: Dict) -> DF:
        print('Nothing to publish because we are running in component test mode.')

    def assert_context(self, ctx: Context) -> NoReturn:
        assert ctx.total_gws == 38
        assert ctx.next_gw == 4
        assert ctx.next_gw_counts['Next 34 GWs'] == 34

    def assert_team_goal_stats_ext(self, team_goal_stats_ext: DF) -> NoReturn:
        team_goal_stats_ext.to_csv(f'{self.tests_path}/team_goal_stats_ext_actual.csv')
        team_goal_stats_ext_exp = pd.read_csv(f'{self.tests_path}/team_goal_stats_ext_expected.csv', index_col='Team Code')

        assert_frame_equal(team_goal_stats_ext, team_goal_stats_ext_exp)

    def assert_player_gw_next_eps_ext(self, player_gw_next_eps_ext: DF) -> NoReturn:
        player_gw_next_eps_ext.to_csv(f'{self.tests_path}/player_gw_next_eps_ext_actual.csv')
        player_gw_next_eps_ext_exp = pd.read_csv(f'{self.tests_path}/player_gw_next_eps_ext_expected.csv', index_col='Player Code', parse_dates=['Kick Off Time', 'News Date', 'Team Last Updated', 'Player Last Updated'])

        assert_frame_equal(player_gw_next_eps_ext, player_gw_next_eps_ext_exp, check_column_type=False)
