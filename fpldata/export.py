import datetime as dt
import pandas as pd
from pathlib import Path
from .common import Context, remove_temp_cols

# Define type aliases
DF = pd.DataFrame

DATA_EXPORT_FORMAT = '%Y-%m-%d %H:%M:%S'
VERSION = '1'


# noinspection PyTypeChecker
def export_df(df: DF, data_dir: str, df_name: str, ctx: Context) -> None:
    # Remove temporary columns.
    df = df.pipe(remove_temp_cols, ctx)

    # Exports the data frame.
    df.to_csv(f'{data_dir}/{df_name}.csv', date_format=DATA_EXPORT_FORMAT)

    # Exports the data dictionary for the data frame.
    (ctx.dd.df(any_data_set=True)
     [lambda dd: dd['Name'].isin(df.reset_index().columns)]
     .to_csv(f'{data_dir}/{df_name}_data_dictionary.csv', date_format=DATA_EXPORT_FORMAT))


def export_dfs(dfs: dict, data_sets: DF, data_dir: str, ctx: Context) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for index, _ in data_sets.iterrows():
        dfs[index].pipe(export_df, data_dir, index, ctx)


def add_data_sets_stats(data_sets: DF, dfs: dict) -> DF:
    for index, _ in data_sets.iterrows():
        df = dfs[index]

        rows = df.shape[0]
        cols = df.reset_index().shape[1]
        data_sets.at[index, 'Data Points'] = rows * cols
        data_sets.at[index, 'Row Count'] = rows
        data_sets.at[index, 'Column Count'] = cols
        data_sets.at[index, 'Last Updated'] = dt.datetime.now()

    return data_sets


def export_data_sets(data_sets: DF, data_dir: str, data_sets_file: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    data_sets.to_csv(f'{data_dir}/{data_sets_file}', date_format=DATA_EXPORT_FORMAT)
