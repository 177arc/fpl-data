![](https://github.com/177arc/fpl-data/workflows/CI%2FCD/badge.svg)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)

# AWS lambda function for calculating FPL data statistics
The purpose of this project is provide to an [AWS lambda function](https://aws.amazon.com/lambda/) that:
1. retrieves data from the [FPL API](https://fpl.readthedocs.io/en/latest/)
2. calculates various statistics, including expected points for each game week, using the [prep-data.ipynb Jupyter notebook](https://github.com/177arc/fpl-data/blob/develop/prep_data.ipynb)
3. makes the prepared data sets available for data analysers
such as the [FPL Advisor](https://github.com/177arc/fpl-advisor). The data sets are published in the public [fpl.177arc.net S3 bucket](http://fpl.177arc.net.s3.eu-west-2.amazonaws.com/list.html)

The lambda function runs in AWS on an hourly schedule during the day and continously updates the data.

# Expected points calculation methodology
The fundamental idea is that the best evidence for a player's ability to generate points is to look over
a sliding window of past fixtures while taking into account the difficulty of the opposing team.

The expected points for each game week for each player are calculated by taking the average points earned by  
each player for every event type (e.g. goals scored, goals conceded, clean sheets, yellow cards, etc.) separately
over a sliding window of past fixtures (currently 12). These averages are adjusted based on the relative strength
of the opposing team compared to the relative strength of the opposing teams that the player has played so far.

# Important data points
The following data points are worth highlighting:
- *Expected Points Next GW* in [player_gw_next_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/player_gw_next_eps_ext.csv): Points that each player is expected to earn in the upcoming game week.
- *Expected Points* in [players_gw_team_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_gw_team_eps_ext.csv): Points that the player is expected to earn for each game week.
- *Expected Goals For* in [players_gw_team_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_gw_team_eps_ext.csv): Goals that the team of the player is expected to score for each game week.
- *Expected Goals Against* in [players_gw_team_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_gw_team_eps_ext.csv): Goals that the team of the player is expected to conceded for each game week.

# List of data sets and data dictionaries
* [player_gw_next_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/player_gw_next_eps_ext.csv) (~120,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/player_gw_next_eps_ext_data_dictionary.csv)):
Contains a row for each player in the current season with expected points for the next game week up to the last one. The data is indexed by the player code which is unique across season.
* [players_gw_team_eps_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_gw_team_eps_ext.csv) (~7,000,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_gw_team_eps_ext_data_dictionary.csv)):
Contains a row for each player and game week combination for the current and last season with the expected points for past and upcoming game weeks. The data is indexed by the player code, the season and the game week number.
* [team_fixture_stats_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/team_fixture_stats_ext.csv) (~100,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/team_fixture_stats_ext_data_dictionary.csv)):
Contains a row for each fixture with the corresponing team info. It has stats for each fixture that are possible indicators of the outcome. These stats are eventually used in the calculation of the expected points. The data is index by the fixture code that is unique across different seasons.
* [players_history_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_history_ext.csv) (~70,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_history_ext_data_dictionary.csv)):
Contains a row for each player fixture combination for the current and the last season with most attributes published by this FPL API endpoint: [](https://fantasy.premierleague.com/api/element-summary/1/). The data is index by the player code and the fixture code, both of them are unique across seasons.
* [fixtures_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/fixtures_ext.csv) (~12,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/fixtures_ext_data_dictionary.csv)):
Contains a row for each fixture in the current and the last season with most attributes published by this FPL API endpoint: [](https://fantasy.premierleague.com/api/fixtures/). The data is indexed by the fixture code that is unique across different seasons.
* [player_teams.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/player_teams.csv) (~36,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/player_teams_data_dictionary.csv)):
Contains a row for each player in the current season with the corresponding team info. The data is index by the player code that is unique across seasons.
* [teams.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/teams.csv) (120 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/teams_data_dictionary.csv)):
Contains a row for each team playing in the current season with most attributes published by this FPL API endpoint: [](https://fantasy.premierleague.com/api/bootstrap-static/). The data is indexed by the team code that is unique across different seasons.
* [players_ext.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_ext.csv) (~42,000 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/players_ext_data_dictionary.csv)):
Contains a row for each player in the current and last season with most of the attributes published by this FPL API endpoint: [](https://fantasy.premierleague.com/api/bootstrap-static/). The data is indexed by the player code that is unique across seasons.
* [gws.csv](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/gws.csv) (646 data points, [data dictionary](https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/v1/latest/gws_data_dictionary.csv)):
Contains a row for each game week of the current season wth most of the game week attributes published by this FPL API endpoint: [](https://fantasy.premierleague.com/api/bootstrap-static/). The data is indexed by the game week ID.