"""
This module contains shared helper functions.
"""

import pandas as pd
import numpy as np
from datadict import DataDict


# Define type aliases
DF = pd.DataFrame
S = pd.Series


class Context:
    POSITION_BY_TYPE: dict = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    FIXTURE_TYPES: list = ['Home', 'Away', '']
    STATS_TYPES: list = ['Goals Scored', 'Goals Conceded', 'Clean Sheets']
    FIXTURE_STATS_TYPES: list = [STATS_TYPES, FIXTURE_TYPES]
    LOCAL_COL_PREFIX = '_'
    PRIVATE_COL_PREFIX = '__'

    total_gws: int                      # The number game weeks in a season.
    next_gw: int                        # The upcoming game week.
    def_next_gws: str                   # The default forecast time horizon, e.g. 'Next 8 GWs'
    next_gw_counts: dict                # The map of time horizon to the number of game weeks, e.g. 'Next 8 GWs' is mapped to 8.
    fixtures_look_back: int             # The rolling number of past fixtures to consider when calculating the fixture stats.
    player_fixtures_look_back: int      # The rolling number of past fixtures to consider when calculating the player stats, in particular the expected points.
    last_season: str                    # The name of the last season, e.g. '2019-20'.
    current_season: str                 # The name of the current season, e.g. '2020-21'.
    dd: DataDict                        # The data dictionary to use for column remapping, formatting and descriptions.


def validate_df(df: DF, df_name: str, required_columns: list):
    """
    Validates that the given data frame has at least certain columns.
    Args:
        df: The data frame to be validated.
        df_name: The name of the data frame for the error message.
        required_columns:

    Raises:
        ValueError: Thrown if the validation fails.
    """
    if not set(df.columns) >= set(required_columns):
        raise ValueError(
            f'{df_name} must at least include the following columns: {required_columns},  {list(set(required_columns) - set(df.columns))} are missing. Please ensure the data frame contains these columns.')


def last_or_default(series: S, default=np.nan):
    """
    Returns the last non-null element of the given series. If non found, returns the default.
    Args:
        series: The series.
        default: The default value.

    Returns:
        Returns the last non-null element of the given series. If non found, returns the default.
    """
    if series is None:
        return default

    series = series[~series.isnull()]
    if series.shape[0] == 0:
        return default

    return series.iloc[-1]


def value_or_default(value, default=np.nan):
    """
    Returns the given value if it is not none. Otherwise returns the default value.

    Args:
        value: The value.
        default: The default.

    Returns:
        Returns the given value if it is not none. Otherwise returns the default value.
    """
    # noinspection PyTypeChecker
    return default if value is None or isinstance(value, str) and value == '' or not isinstance(value, str) and np.isnan(value) else value


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


def remove_temp_cols(df: DF, ctx: Context) -> DF:
    return df[[col for col in df.columns if not col.startswith(ctx.LOCAL_COL_PREFIX)]]
