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
            .reset_index()
            [lambda df: (df['Fixture Minutes Played'] > 0) & (df['Fixtures Played Recent Fixtures'] > 4) &
                        ((df['Season'] == ctx.current_season) & (df['Game Week'] < ctx.next_gw) | (df['Season'] != ctx.current_season))]
            .assign(**{'Player Fixture Error': lambda df: np.abs(df['Expected Points'] - df['Fixture Total Points'])})
            .assign(**{'Player Fixture Error Simple': lambda df: np.abs(df['Expected Points Simple'] - df['Fixture Total Points'])})
            .assign(**{'Player Fixture Sq Error': lambda df: df['Player Fixture Error'] ** 2})
            .assign(**{'Player Fixture Sq Error Simple': lambda df: df['Player Fixture Error Simple'] ** 2})
            .groupby(['Season', 'Game Week'])
            [['Player Fixture Error', 'Player Fixture Sq Error', 'Expected Points', 'Player Fixture Error Simple', 'Player Fixture Sq Error Simple', 'Expected Points Simple', 'Fixture Total Points', 'Player Code']]
            .agg({'Player Fixture Error': 'sum', 'Player Fixture Sq Error': 'sum', 'Expected Points': 'sum', 'Player Fixture Error Simple': 'sum', 'Player Fixture Sq Error Simple': 'sum', 'Expected Points Simple': 'sum',
                  'Fixture Total Points': 'sum', 'Player Code': 'count'})
            .reset_index()
            .pipe(add_gws_ago)
            [lambda df: df['GWs Ago'] <= ctx.fixtures_look_back]
            .sort_values(['Season', 'Game Week'])
            .rename(columns={'Player Code': 'Player Count'})
            .assign(**{'Avg Expected Points': lambda df: df['Expected Points'] / df['Player Count']})
            .assign(**{'Avg Expected Points Simple': lambda df: df['Expected Points Simple'] / df['Player Count']})
            .assign(**{'Avg Fixture Total Points': lambda df: df['Fixture Total Points'] / df['Player Count']})
            .assign(**{'Error': lambda df: df['Player Fixture Error'] / df['Player Count']})
            .assign(**{'Error Simple': lambda df: df['Player Fixture Error Simple'] / df['Player Count']})
            .assign(**{'Sq Error': lambda df: df['Player Fixture Sq Error'] / df['Player Count']})
            .assign(**{'Sq Error Simple': lambda df: df['Player Fixture Sq Error Simple'] / df['Player Count']})
            .assign(**{'Season Game Week': lambda df: df['Season'] + ', GW ' + df['Game Week'].apply('{:.0f}'.format)}))
