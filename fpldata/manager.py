from typing import Tuple, NoReturn, Dict
from shutil import copyfile, rmtree
from fplpandas import FPLPandas
import logging
import tempfile
import pandas as pd
import datetime as dt
from datadict import DataDict
import numpy as np

from .s3store import S3Store
from .export import export_dfs, add_data_sets_stats, export_data_sets, VERSION
from .common import Context

# Define type aliases
DF = pd.DataFrame
S = pd.Series


class FPLManagerBase:
    mode: str

    def create_context(self) -> Context:
        raise NotImplementedError

    def get_game_weeks(self) -> DF:
        raise NotImplementedError

    def get_teams(self) -> DF:
        raise NotImplementedError

    def get_teams_last_season(self) -> DF:
        raise NotImplementedError

    def get_fixtures(self) -> DF:
        raise NotImplementedError

    def get_fixtures_last_season(self) -> DF:
        raise NotImplementedError

    def get_players(self) -> Tuple[DF, DF, DF]:
        raise NotImplementedError

    def get_players_last_season(self) -> Tuple[DF, DF, DF]:
        raise NotImplementedError

    def get_last_season_stats_est(self) -> DF:
        raise NotImplementedError

    def publish_data_sets(self, variables: Dict) -> DF:
        raise NotImplementedError

    def assert_context(self, ctx: Context) -> NoReturn:
        raise NotImplementedError

    def assert_team_goal_stats_ext(self, team_goal_stats_ext: DF) -> NoReturn:
        raise NotImplementedError

    def assert_player_gw_next_eps_ext(self, player_gw_next_eps_ext: DF) -> NoReturn:
        raise NotImplementedError

    def assert_players_gw_team_eps_ext(self, players_gw_team_eps_ext: DF) -> NoReturn:
        raise NotImplementedError


class FPLManager(FPLPandas, FPLManagerBase):
    TEAMS_FILE = 'teams.csv'  # File with team data for last season
    PLAYERS_FILE = f'players.csv'  # File with player data for last season
    PLAYERS_HISTORY_FILE = f'players_history.csv'  # File with player fixture data for last season
    FIXTURES_FILE = f'fixtures.csv'  # File with fixture data for last season
    TEAM_STATS_EST_FILE = 'data/team_goals_stats_estimates.csv'  # Path to file containing goal estimates for team that just joined the league
    DATA_TEMP_DIR = f'{tempfile.gettempdir()}/fpl_data'
    DATA_DIR = f'data'
    DATA_SETS_FILE = f'data/data_sets.csv'
    DATA_DICT_FILE = f'data/data_dictionary.csv'
    DEF_FIXTURE_LOOK_BACK = 20  # Limit of how many fixtures to look back for calculating rolling team stats
    DEF_PLAYER_FIXTURE_LOOK_BACK = 12  # Limit of how many fixture to look back for calculating rolling player stats

    last_season: str
    current_season: str
    last_season_path: str
    fixtures_look_back: int
    player_fixtures_look_back: int

    publish_s3_bucket: str

    def __init__(self, last_season: str, current_season: str, publish_s3_bucket: str,
                 fixtures_look_back: int = DEF_FIXTURE_LOOK_BACK, player_fixtures_look_back: int = DEF_PLAYER_FIXTURE_LOOK_BACK):
        self.last_season = last_season
        self.current_season = current_season
        self.last_season_path = f'{self.DATA_DIR}/{last_season}'
        self.fixtures_look_back = fixtures_look_back
        self.player_fixtures_look_back = player_fixtures_look_back
        self.publish_s3_bucket = publish_s3_bucket
        self.mode = 'Live'

        super().__init__()

    def create_context(self) -> Context:
        ctx = Context()
        ctx.fixtures_look_back = self.fixtures_look_back
        ctx.player_fixtures_look_back = self.player_fixtures_look_back
        ctx.last_season = self.last_season
        ctx.current_season = self.current_season
        ctx.now = dt.datetime.now()
        return ctx

    def get_teams_last_season(self) -> DF:
        return pd.read_csv(f'{self.last_season_path}/{self.TEAMS_FILE}', index_col=['id'], na_values='None')

    def get_fixtures_last_season(self) -> DF:
        return pd.read_csv(f'{self.last_season_path}/{self.FIXTURES_FILE}', index_col=['id'], na_values='None')

    def get_players_last_season(self) -> Tuple[DF, DF, DF]:
        return (pd.read_csv(f'{self.last_season_path}/{self.PLAYERS_FILE}', index_col=['id'], na_values='None'),
                None,
                pd.read_csv(f'{self.last_season_path}/{self.PLAYERS_HISTORY_FILE}', index_col=['player_id', 'fixture'], na_values='None'))

    def get_last_season_stats_est(self) -> DF:
        # Loads estimates for team for which no history is available, in particular for those that have been promoted to the Premier League.
        return pd.read_csv(self.TEAM_STATS_EST_FILE).set_index('Team Code')

    def publish_data_sets(self, variables: Dict) -> DF:
        logging.info(f'Publishing data sets to {self.publish_s3_bucket}/v{VERSION}/latest/ ...')

        s3store = S3Store(self.publish_s3_bucket)

        # Clear the data directory
        rmtree(self.DATA_TEMP_DIR, ignore_errors=True)

        data_sets = pd.read_csv(self.DATA_SETS_FILE).set_index('Name')

        (data_sets
         .pipe(add_data_sets_stats, variables)
         .pipe(export_data_sets, f'{self.DATA_TEMP_DIR}/v{VERSION}', self.DATA_SETS_FILE.split("/")[-1]))

        # Export data frames as CSV files.
        export_dfs(variables, data_sets, f'{self.DATA_TEMP_DIR}/v{VERSION}', DataDict(data_dict_file=self.DATA_DICT_FILE))

        # Copy the data dictionary and data sets file.
        _ = copyfile(self.DATA_DICT_FILE, f'{self.DATA_TEMP_DIR}/v{VERSION}/{self.DATA_DICT_FILE.split("/")[-1]}')

        # And off we go to S3.
        s3store.save_dir(self.DATA_TEMP_DIR, f'v{VERSION}/latest/')

        logging.info('Done!')

    def assert_context(self, ctx: Context) -> NoReturn:
        assert ctx.next_gw + len(ctx.next_gw_counts.keys()) == ctx.total_gws

    def assert_team_goal_stats_ext(self, team_goal_stats_ext: DF) -> NoReturn:
        # TODO: Implement some sense checks.
        pass

    def assert_player_gw_next_eps_ext(self, player_gw_next_eps_ext: DF) -> NoReturn:
        assert player_gw_next_eps_ext[lambda df: df.isin([np.inf, -np.inf]).any(axis=1)].shape[0] == 0, 'There are inifinite values in player_gw_next_eps_ext. Run player_gw_next_eps_ext[lambda df: df.isin([np.inf, -np.inf]).any(axis=1)] to finds row with inifite values.'

    def assert_players_gw_team_eps_ext(self, players_gw_team_eps_ext: DF) -> NoReturn:
        assert players_gw_team_eps_ext[lambda df: df.isin([np.inf, -np.inf]).any(axis=1)].shape[0] == 0, 'There are inifinite values in players_gw_team_eps_ext. Run players_gw_team_eps_ext[lambda df: df.isin([np.inf, -np.inf]).any(axis=1)] to finds row with inifite values.'
