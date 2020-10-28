"""
This module contains functions for back testing the expected points predictions.
"""

import pandas as pd
import numpy as np
from .common import Context


# Define type aliases
DF = pd.DataFrame
S = pd.Series

def add_gws_ago(players_gw_team: DF) -> DF:
    return (players_gw_team
            .sort_values(['Season', 'Game Week'], ascending=False)
            .assign(**{'GWs Ago': lambda df: (~df['Expected Points'].isnull()).cumsum()}))

def get_gw_points_backtest(players_gw_team_eps: DF, ctx: Context) -> DF:
    return (players_gw_team_eps
         [lambda df: df['Fixture Minutes Played'] > 0]
         .reset_index()
         .groupby(['Season', 'Game Week'])
         [['Expected Points', 'Expected Points Simple', 'Fixture Total Points', 'Player Code']]
         .agg({'Expected Points': 'sum', 'Expected Points Simple': 'sum', 'Fixture Total Points': 'sum', 'Player Code': 'count'})
         .reset_index()
         .pipe(add_gws_ago)
         [lambda df: df['GWs Ago'] <= ctx.fixtures_look_back]
         .sort_values(['Season', 'Game Week'])
         .rename(columns={'Player Code': 'Player Count'})
         .assign(**{'Avg Expected Points': lambda df: df['Expected Points']/df['Player Count']})
         .assign(**{'Avg Fixture Total Points': lambda df: df['Fixture Total Points']/df['Player Count']})
         .assign(**{'Avg Expected Points Simple': lambda df: df['Expected Points Simple']/df['Player Count']})
         .assign(**{'Error': lambda df: np.abs(df['Avg Expected Points']-df['Avg Fixture Total Points'])})
         .assign(**{'Error Simple': lambda df: np.abs(df['Avg Expected Points Simple']-df['Avg Fixture Total Points'])})
         .assign(**{'Season Game Week': lambda df: df['Season']+', GW '+df['Game Week'].apply('{:.0f}'.format)}))
