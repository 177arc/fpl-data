import pandas as pd
from .common import FPL, Context
from typing import Tuple, NoReturn

#Define type aliases
DF = pd.DataFrame

class FPLTest(FPL):
    tests_path: str

    def __init__(self, tests_path):
        self.tests_path = tests_path
        super().__init__()

    def get_game_weeks(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/test_gws.csv', index_col=['id'], na_values='None')

    def get_teams(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/test_teams.csv', index_col=['id'], na_values='None')

    def get_teams_last_season(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/test_teams_last_season.csv', index_col=['id'], na_values='None')

    def get_fixtures(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/test_fixtures.csv', index_col=['id'], na_values='None')

    def get_fixtures_last_season(self) -> DF:
        return pd.read_csv(f'{self.tests_path}/test_fixtures_last_season.csv', index_col=['id'], na_values='None')

    def get_players(self) -> Tuple[DF, DF, DF]:
        return (pd.read_csv(f'{self.tests_path}/test_players.csv', index_col=['id'], na_values='None'),
                None,
                pd.read_csv(f'{self.tests_path}/test_players_history.csv', index_col=['player_id', 'fixture'], na_values='None'))

    def get_players_last_season(self) -> Tuple[DF, DF, DF]:
        return (pd.read_csv(f'{self.tests_path}/test_players_last_season.csv', index_col=['id'], na_values='None'),
                None,
                pd.read_csv(f'{self.tests_path}/test_players_history_last_season.csv', index_col=['player_id', 'fixture'], na_values='None'))

    def assert_context(ctx: Context) -> NoReturn:
        assert ctx.total_gws == 38
        assert ctx.next_gw == 3
        assert ctx.next_gw_counts['Next 35 GWs'] == 35