#!/usr/bin/env python
# coding: utf-8

# # Installation
# To get started, run the following command to install all required dependencies.

# In[1]:


#!pip install -q -r ./requirements.txt


# # Import requirements
# Here we import all external and local modulues.

# In[2]:


from common import *
from derive import *
from s3store import *
from fplpandas import FPLPandas
from pathlib import Path
from shutil import copyfile
import tempfile
import logging

# Define type aliases
DF = pd.DataFrame
S = pd.Series


# # Set variables
# This section sets all important global variables.

# In[3]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False


# In[4]:


temp_dir = tempfile.gettempdir()
data_dict_file = 'data/data_dictionary.csv'
data_dir = f'{temp_dir}/data'

logging.basicConfig(level=(logging.WARN if is_notebook() else logging.INFO))

ctx = Context()
ctx.fixtures_look_back = 38  # Limit of how many fixtures to look back for calculating rolling team stats
ctx.player_fixtures_look_back = 10 # Limit of how many fixture to look back for calcating rolling player stats
ctx.last_season = '2019-20'
ctx.current_season = '2020-21'

players_path = f'{ctx.last_season}/players.csv'
players_history_path = f'{ctx.last_season}/players_history.csv'
fixtures_path = f'{ctx.last_season}/fixtures.csv'
teams_path = f'{ctx.last_season}/teams.csv'

fpl = FPLPandas() # Wrapper for access the FPL API and mapping the data into pandas data frames.


# # Load data dictionary
# This section loads the data dictionary. The data dictionary contains default ordering of fields, for each field a description, default format and mapping of API field names to more readable ones. It is used to show data in a more user-friendly way.

# In[5]:


ctx.dd = DataDict(data_dict_file=data_dict_file)


# # Load game week data
# The data frame contains one row for each game week for the current season.

# In[6]:


logging.info('Loading game week data ...')

gws = get_game_weeks(fpl, ctx)
ctx.total_gws = gws.shape[0]
ctx.next_gw = gws[lambda df: df['Is Next GW?']].index.values[0]
ctx.def_next_gws = 'Next 8 GWs'
ctx.next_gw_counts = get_next_gw_counts(ctx)


# # Load team data
# This section loads the team data and stats from the following endpoint: https://fantasy.premierleague.com/api/bootstrap-static/ and returns it as a panda data frame.

# In[7]:


logging.info('Loading team data ...')

# Get current team data. The resultnig data frame contains one row for each team playing in the current season.
teams = fpl.get_teams().pipe(prepare_teams, ctx)
team_id_code_map = teams.pipe(get_team_id_code_map)


# In[8]:


# Get last season's team data
teams_last_season = (pd.read_csv(teams_path, index_col=['id']).pipe(prepare_teams, ctx))
teams_last_season_id_code_map = teams_last_season.pipe(get_team_id_code_map)

# This data frame contains one row for each team playing in the current season and the past season.
teams_ext = pd.concat([teams, teams_last_season[~teams_last_season.index.isin(teams.index)]])


# # Load fixture data
# This section loads the fixture data and stats from the following endpoint: https://fantasy.premierleague.com/api/fixtures/ and returns it as a panda data frame.

# In[9]:


logging.info('Loading fixture data ...')

# Get current fixture data. The resulting data frame contains one row for each fixture of the current season, both past and future ones.
fixtures = fpl.get_fixtures().pipe(prepare_fixtures, team_id_code_map, ctx).assign(**{'Season': ctx.current_season})
fixtures_id_code_map = fixtures.pipe(get_fixtures_id_code_map)


# In[10]:


# Get last season's fixture data
fixtures_last_season = (pd.read_csv(fixtures_path, index_col=['id'])
                        .pipe(prepare_fixtures, teams_last_season_id_code_map, ctx)).assign(**{'Season': ctx.last_season})
fixtures_last_season_id_code_map = fixtures_last_season.pipe(get_fixtures_id_code_map)

# This data frame contains one row for each fixture of the the last and the current season, both past and future ones.
fixtures_ext = pd.concat([fixtures, fixtures_last_season])


# # Load player data
# This section loads the player data and stats from the following FPL API endpoint: https://fantasy.premierleague.com/api/bootstrap-static/ and returns it as a panda data frame. **This can take a few seconds** because for each player the full history for the current season is downloaded.

# In[11]:


logging.info('Loading player data ...')

# Get current player data
players = (get_players_raw(fpl)[0]
           .pipe(prepare_players, ctx)
           .assign(**{'Season': ctx.current_season}))
players_id_code_map = players.pipe(get_players_id_code_map)
players_raw = get_players_raw(fpl)

# This data frame contains one row for every past season played in the premier league for every player in the current season.
players_history_past = players_raw[1].pipe(prepare_players_history_past, players_id_code_map, ctx)

# This data frame contains one row for every completed fixture in the current season for every player.
players_history = (players_raw[2]
                   .pipe(prepare_players_history, players_id_code_map, fixtures_id_code_map, ctx)
                   .assign(**{'Season': ctx.current_season}))


# In[12]:


# Get last season's player data
players_last_season =  (pd.read_csv(players_path, index_col=['id'], na_values='None')
                        .pipe(prepare_players, ctx).assign(**{'Season': ctx.last_season}))
players_last_season_id_code_map = players_last_season.pipe(get_players_id_code_map)

players_history_last_season = (pd.read_csv(players_history_path, index_col=['player_id', 'fixture'])
            .pipe(prepare_players_history, players_last_season_id_code_map, fixtures_last_season_id_code_map, ctx)
            .assign(**{'Season': ctx.last_season}))

# This data frame contains one row for every player in the last and the current season.
players_ext = pd.concat([players, players_last_season[~players_last_season.index.isin(players.index)]])

# This data frame contains one row for every completed fixuture in the last and the current season.
players_history_ext = pd.concat([players_history, players_history_last_season])


# # Create derived data
# This section creates new dataset by combining the previously loaded ones.

# In[13]:


logging.info('Creating derived data sets ...')


# ## Players with team info

# In[14]:


player_teams = players.pipe(get_player_teams, teams, ctx)
player_teams_ext = players.pipe(get_player_teams, teams_ext, ctx)


# ## Fixtures with team info

# In[15]:


fixture_teams_ext = fixtures_ext.pipe(get_fixture_teams, teams_ext, ctx)


# ## Player derived fields and metrics
# The section below derives a few useful player attributes but most importantly, it calculates the total points earned by a player devided by his current cost. This is can be an indicator for whether the player is undervalued or overpriced.

# In[16]:


players_history_fixtures_ext = players_history_ext.pipe(get_players_history_fixtures, fixtures_ext, player_teams_ext, ctx)


# ## Team metrics

# In[17]:


team_score_stats_est = load_team_goal_stats_est('team_goals_stats_estimates.csv', ctx)
team_score_stats_ext = fixture_teams_ext.pipe(get_team_score_stats, teams_ext, team_score_stats_est, ctx)


# ## Fixture stats
# In order to calculate relative strengths of the teams, we aggregate the points that the team has earned so far. We later can use this information to adjust the expected points for each player.

# In[18]:


fixture_teams_stats_ext = fixture_teams_ext.pipe(get_fixture_teams_stats, team_score_stats_ext, ctx)


# ## Calculate relative fixture strengths
# Calculates a relative fixtures strengths for each team. The relative strength is a factor around 1 and is used in the expected point prediction below to adjust the predicted points based on the relative strengths of the upcoming game weeks. The simple idea here is that team with more total points so far are stronger. A value above 1 indicates that the player's team is relatively stronger and a value below 1 indicates that the team is relatively weaker. 

# In[19]:


team_fixture_strength_ext = fixture_teams_stats_ext.pipe(get_team_fixture_strength, teams_ext, ctx)


# ## Transfer relative fixture strengths from fixtures to players
# This section joins the fixture strengths data set with the player data set so that expected points can be calculated on a player basis.

# In[20]:


player_team_fixture_strength_ext = players.pipe(get_player_team_fixture_strength, team_fixture_strength_ext, players_history_ext, ctx)


# ## Create combined data for past and future fixtures for each player
# This section concatenates two sets: one historical and one future fixture set. The reason for this is that for completed matches, we need it to consider the team that player actually played for, while for future games we can assume that the player will play for the same team than he is currently in.

# In[21]:


team_future_fixtures = get_team_future_fixtures(team_fixture_strength_ext, players_history_fixtures_ext)
players_future_fixture_team_strengths = get_players_future_fixture_team_strengths(player_teams, team_future_fixtures)
player_fixture_stats = get_player_fixture_stats(players_history_fixtures_ext, players_future_fixture_team_strengths, player_team_fixture_strength_ext)


# ## Calculates the expected points for the following time horizons
# Calculates the cumulative expected points for the all the game weeks up to the end of the season. The expected points for each time horizon are simply the sum of expected points for each game week within the time horizon.

# Calculate expected points for each player and fixture combination.

# In[22]:


players_fixture_team_eps_ext = get_players_fixture_team_eps(player_fixture_stats)


# Project the fixtures to game week level to deal with game weeks when there is no fixture for a team or double fixtures.

# In[23]:


players_gw_team_eps_ext = get_players_gw_team_eps(players_fixture_team_eps_ext, player_teams)


# Calculates the expected points for the different time horizons for each player.

# In[24]:


player_gw_next_eps_ext = get_player_gw_next_eps(players_gw_team_eps_ext, ctx)


# # Publish Data Sets in S3

# In[25]:


logging.info('Publishing data sets to S3 ...')

s3store = S3Store()


# In[26]:


dfs = {'gws': gws,
               'players_ext': players_ext,
               'teams': teams,
               'player_teams': player_teams,
               'fixtures_ext': fixtures_ext,
               'players_history_ext': players_history_ext,
               'team_fixture_strength_ext': team_fixture_strength_ext,
               'players_gw_team_eps_ext': players_gw_team_eps_ext,
               'player_gw_next_eps_ext': player_gw_next_eps_ext
              }


# In[28]:


def export_dfs(dfs: Dict[str, DF], data_dir: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for df_name, df in dfs.items():
        df.to_csv(f'{data_dir}/{df_name}.csv')

# Export data frames as CSV files.
export_dfs(dfs, data_dir)

# Copy the data dictory.
_ = copyfile(data_dict_file, f'{data_dir}/{data_dict_file}')

# And off we go to S3.
s3store.save_dir(data_dir)

logging.info('Done!')


# In[ ]:




