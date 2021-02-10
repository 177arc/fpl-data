import unittest
import boto3
import warnings
import logging
import io

import botocore
import pandas as pd
import requests
from typing import Union, List

# Define type aliases
DF = pd.DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestE2E(unittest.TestCase):
    def __get_df(self, url: str, index: Union[str, List[str]] = None) -> DF:
        res = requests.get(url)
        df = pd.read_csv(io.BytesIO(res.content))

        if index is not None:
            df.set_index(index, drop=True)

        return df

    def setUp(self) -> None:
        warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<ssl.SSLSocket.*>')

        self.BUCKET = 'fpl-test.177arc.net'
        self.LAMBDA_NAME = 'fpl-data-test'
        self.REGION_NAME = 'eu-west-2'
        self.DATA_URL = f'https://s3.{self.REGION_NAME}.amazonaws.com/{self.BUCKET}/'

        cfg = botocore.config.Config(retries={'max_attempts': 0}, read_timeout=300, connect_timeout=300)
        self.lmda = boto3.client('lambda', region_name=self.REGION_NAME, config=cfg)
        self.s3 = boto3.client('s3')


    def test_lambda(self) -> None:
        dfs = [
            dict(file='v1/latest/gws.csv', index='GW ID'),
            dict(file='v1/latest/teams.csv', index='Team Code'),
            dict(file='v1/latest/players_ext.csv', index='Player Code'),
            dict(file='v1/latest/player_teams.csv', index='Player Code'),
            dict(file='v1/latest/players_gw_team_eps_ext.csv', index=['Player Code', 'Season', 'Game Week']),
            dict(file='v1/latest/player_gw_next_eps_ext.csv', index='Player Code'),
            dict(file='v1/latest/team_fixture_stats_ext.csv', index='Team Code'),
            dict(file='v1/latest/data_dictionary.csv', index=None),
            dict(file='v1/latest/data_sets.csv', index=None),
        ]

        # Delete previous artifacts
        logging.info(f'Deleting previous artifacts ...')
        for df in dfs:
            self.s3.delete_object(Bucket=self.BUCKET, Key=df['file'])

        # Invoke lambda
        logging.info(f'Invoking {self.LAMBDA_NAME} ...')
        response = self.lmda.invoke(FunctionName=self.LAMBDA_NAME, InvocationType='RequestResponse', LogType='Tail')

        logging.info(response['LogResult'])

        # Check response
        self.assertEqual(response['StatusCode'], 200)
        self.assertFalse('FunctionError' in response)

        # Check data
        for df in dfs:
            self.assertTrue(self.__get_df(url=f'{self.DATA_URL}{df["file"]}', index=df['index']).shape[0] > 0)
