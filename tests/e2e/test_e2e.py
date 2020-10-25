import unittest
import boto3
import warnings
import logging
import io
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

        self.bucket = 'fpl-test.177arc.net'
        self.lambda_name = 'fpl-data-test'
        self.region_name = 'eu-west-2'
        self.data_url = f'https://s3.{self.region_name}.amazonaws.com/{self.bucket}/'

        self.lmda = boto3.client('lambda', region_name=self.region_name)
        self.s3 = boto3.client('s3')


    def test_lambda(self) -> None:
        dfs = [
            dict(file='gws.csv', index='GW ID'),
            dict(file='teams.csv', index='Team Code'),
            dict(file='players_ext.csv', index='Player Code'),
            dict(file='player_teams.csv', index='Player Code'),
            dict(file='players_gw_team_eps_ext.csv', index=['Player Code', 'Season', 'Game Week']),
            dict(file='player_gw_next_eps_ext.csv', index='Player Code'),
            dict(file='team_fixture_strength_ext.csv', index='Team Code'),
            dict(file='data_dictionary.csv', index=None),
        ]

        # Delete previous artifacts
        logging.info(f'Deleting previous artifacts ...')
        for df in dfs:
            self.s3.delete_object(Bucket=self.bucket, Key=df['file'])

        # Invoke lambda
        logging.info(f'Invoking {self.lambda_name} ...')
        response = self.lmda.invoke(FunctionName=self.lambda_name, InvocationType='RequestResponse', LogType='Tail')

        # Check response
        self.assertEqual(response['StatusCode'], 200)
        self.assertFalse('FunctionError' in response)

        logging.info(response['LogResult'])

        # Check data
        for df in dfs:
            self.assertTrue(self.__get_df(url=f'{self.data_url}{df["file"]}', index=df['index']).shape[0] > 0)
