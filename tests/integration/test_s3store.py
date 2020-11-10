import unittest
import numpy as np
import pandas as pd
import warnings
from pandas.testing import assert_frame_equal
from typing import Dict
from io import BytesIO, StringIO
from zipfile import ZipFile, ZIP_DEFLATED
import sys
import os
import boto3

# Define type aliases
DF = pd.DataFrame

# Add project root directory to Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from fpldata.s3store import S3Store

class TestS3Store(unittest.TestCase):
    """
    Tests the class that persists data frames to S3. To be able to execute these integration tests,
    you need:
        a) an AWS account
        b) test BUCKET (Note you will need to update the test_s3_bucket variable accordingly)
        c) your AWS keys locally (see https://docs.aws.amazon.com/AmazonS3/latest/dev/setup-aws-cli.html)
    """

    test_s3_bucket = 'fpl-test.177arc.net'
    test_obj_name = 'test.csv'

    s3store: S3Store
    s3: boto3.resource


    def __read_df(self, obj_name: str) -> DF:
        obj = self.s3.get_object(Bucket=self.test_s3_bucket, Key=obj_name)
        content = BytesIO(obj['Body'].read())

        compression = None
        if obj['ContentEncoding'] == 'gzip':
            compression = 'gzip'

        return pd.read_csv(content, compression=compression)

    def __read_df_zip(self, obj_name: str) -> DF:
        return list(self.__read_dfs_zip(obj_name).values())[0]

    def __read_dfs_zip(self, obj_name: str) -> Dict[str, DF]:
        zf = self.__read_zip(obj_name)
        dfs = {}

        with zf:
            for file_name in zf.namelist():
                dfs[file_name.replace('.csv', '')] = pd.read_csv(BytesIO(zf.read(file_name)))

        return dfs

    def __read_zip(self, obj_name: str) -> ZipFile:
        obj = self.s3.get_object(Bucket=self.test_s3_bucket, Key=obj_name)
        buffer = BytesIO(obj["Body"].read())
        return ZipFile(buffer)

    def __write_df(self, df: DF, obj_name: str) -> None:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        self.s3.put_object(Bucket=self.test_s3_bucket, Key=obj_name, Body=csv_buffer.getvalue())

    def __write_df_zip(self, df: DF, obj_name: str) -> None:
        self.__write_dfs_zip({obj_name.replace('.zip', ''): df}, obj_name)

    def __write_dfs_zip(self, dfs: Dict[str,DF], obj_name: str) -> None:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, mode='w', compression=ZIP_DEFLATED) as zf:
            for df_name, df in dfs.items():
                csv_buffer = StringIO()
                df.to_csv(csv_buffer)
                zf.writestr(df_name+'.csv', csv_buffer.getvalue())

        self.s3.put_object(Bucket=self.test_s3_bucket, Key=obj_name, Body=zip_buffer.getvalue())

    def __del(self, obj_name: str) -> None:
        self.s3.delete_object(Bucket=self.test_s3_bucket, Key=obj_name)

    def setUp(self) -> None:
        warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>")

        self.s3store = S3Store(s3_bucket=self.test_s3_bucket)
        self.s3 = boto3.client('s3')
        self.dfs = {'df1': pd.DataFrame.from_dict(
                        {0: ['test 1', 1, True, 1.1, 'bayern'],
                        1: ['test 2', 2, False, 1.2, 'bayern'],
                        2: [np.nan, 3, np.nan, np.nan, 'bayern']},
                            orient='index',
                            columns=['Name 1', 'Name 2', 'Name 3', 'Name 4', 'Name 5']).set_index('Name 2'),
                    'df2': pd.DataFrame.from_dict(
                    {0: ['test 1', 1, True, 1.1],
                    1: ['test 2', 2, False, 1.2]},
                        orient='index',
                        columns=['Col 1', 'Col 2', 'Col 3', 'Col 4']).set_index('Col 1')}

        self.df = list(self.dfs.values())[0]

    def test_save_df(self) -> None:
        # Set up
        self.__del(self.test_obj_name)

        # Execute test
        self.s3store.save_df(self.df, self.test_obj_name)

        # Assert
        actual_df = self.__read_df(self.test_obj_name)
        assert_frame_equal(self.df.reset_index(), actual_df, check_dtype=False, check_column_type=False)

    def test_save_df_zip(self) -> None:
        # Set up
        self.__del(self.test_obj_name + '.zip')

        # Execute test
        self.s3store.save_df(self.df, self.test_obj_name + '.zip')

        # Assert
        actual_df = self.__read_df_zip(self.test_obj_name + '.zip')
        assert_frame_equal(self.df.reset_index(), actual_df, check_dtype=False, check_column_type=False)

    def test_save_dfs(self) -> None:
        # Set up
        self.__del(self.test_obj_name)

        # Execute test
        self.s3store.save_dfs(self.dfs, self.test_obj_name + '.zip')

        # Assert
        actual_dfs = self.__read_dfs_zip(self.test_obj_name + '.zip')
        for df_name, df in self.dfs.items():
            assert_frame_equal(df.reset_index(), actual_dfs[df_name], check_dtype=False, check_column_type=False)

    def test_save_dir_zip(self):
        # Set up
        dfs = {}
        dfs['df1'] = pd.read_csv('dfs/df1.csv')
        dfs['df2'] = pd.read_csv('dfs/df2.csv')

        # Execute test
        self.s3store.save_dir('dfs', self.test_obj_name + '.zip')

        # Assert
        actual_dfs = self.__read_dfs_zip(self.test_obj_name + '.zip')

        for df_name, df in dfs.items():
            assert_frame_equal(df, actual_dfs[df_name], check_dtype=False, check_column_type=False)

    def test_save_dir(self):
        # Set up
        dfs = {}
        dfs['df1'] = pd.read_csv('dfs/df1.csv')
        dfs['df2'] = pd.read_csv('dfs/df2.csv')

        # Execute test
        self.s3store.save_dir('dfs')

        # Assert
        actual_dfs = {}
        for df_name in dfs.keys():
            actual_dfs[df_name] = self.__read_df(df_name+'.csv')

        for df_name, df in dfs.items():
            assert_frame_equal(df, actual_dfs[df_name], check_dtype=False, check_column_type=False)

    def test_save_dir_key(self):
        # Set up
        dfs = {}
        dfs['df1'] = pd.read_csv('dfs/df1.csv')
        dfs['df2'] = pd.read_csv('dfs/df2.csv')

        # Execute test
        self.s3store.save_dir('dfs', 'dfs/')

        # Assert
        actual_dfs = {}
        for df_name in dfs.keys():
            actual_dfs[df_name] = self.__read_df('dfs/'+df_name+'.csv')

        for df_name, df in dfs.items():
            assert_frame_equal(df, actual_dfs[df_name], check_dtype=False, check_column_type=False)

    def test_save_file_gz(self):
        # Execute test
        self.s3store.save_file('dfs/df1.csv', 'df1.csv', content_encoding='gzip')

        # Assert
        actual_df = self.__read_df('df1.csv')
        assert_frame_equal(pd.read_csv('dfs/df1.csv'), actual_df, check_dtype=False, check_column_type=False)

    def test_save_file(self):
        # Execute test
        self.s3store.save_file('dfs/df1.csv', 'df1.csv')

        # Assert
        actual_df = self.__read_df('df1.csv')
        assert_frame_equal(pd.read_csv('dfs/df1.csv'), actual_df, check_dtype=False, check_column_type=False)

    def test_load_df(self) -> None:
        # Set up
        self.__write_df(self.df, self.test_obj_name)

        # Execute test
        actual_df = self.s3store.load_df(self.test_obj_name)

        # Assert
        assert_frame_equal(self.df.reset_index(), actual_df, check_dtype=False, check_column_type=False)

    def test_load_df_zip(self) -> None:
        # Set up
        self.__write_df_zip(self.df, self.test_obj_name + '.zip')

        # Execute test
        actual_df = self.s3store.load_df(self.test_obj_name + '.zip')

        # Assert
        assert_frame_equal(self.df.reset_index(), actual_df, check_dtype=False, check_column_type=False)

    def test_load_dfs(self) -> None:
        # Set up
        self.__write_dfs_zip(self.dfs, self.test_obj_name + '.zip')

        # Execute test
        actual_dfs = self.s3store.load_dfs(self.test_obj_name + '.zip')

        # Assert
        for df_name, df in self.dfs.items():
            assert_frame_equal(df.reset_index(), actual_dfs[df_name], check_dtype=False, check_column_type=False)

