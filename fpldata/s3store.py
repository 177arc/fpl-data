import boto3
import pandas as pd
from io import StringIO, BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from gzip import GzipFile
import os
from typing import Dict
import shutil

# Define type aliases
DF = pd.DataFrame
S = pd.Series


class S3Store():
    def_s3_bucket = 'fpl.177arc.net'

    def __is_zip(self, file_name: str) -> bool:
        return file_name.endswith('.zip')

    def __is_csv(self, file_name: str) -> bool:
        return file_name.endswith('.csv')

    def __init__(self,  s3_bucket: str = None, s3: boto3.client = None):
        """
        Initialise the S3 store.

        Args:
            s3:  The S3 resource. If not provided, defaults to the standard S3 resource.
            s3_bucket: The name of the S3 bucket. If not provided, default to 'fpl.177arc.net'.
        """
        self.s3 = s3 if s3 is not None else boto3.client('s3')
        self.s3_bucket = s3_bucket if s3_bucket is not None else self.def_s3_bucket

    def save_df(self, df: DF, key_name: str) -> None:
        """
        Saves the given data frame to the S3 bucket.

        Args:
            df: The data frame to be saved.
            key_name: The name of the S3 object. If the key name ends in '.zip', it will generate a zip archive in S3.
        """
        if not self.__is_zip(key_name) and not self.__is_csv(key_name):
            raise ValueError(f'The key name {key_name} does not end in \'.zip\' or \'.csv\'. Please use one of these extension to indicate how the data frame should be saved.')

        self.save_dfs({key_name.replace('.zip', '').replace('.csv', ''): df}, key_name)

    def save_dfs(self, dfs: Dict[str, DF], key_name: str) -> None:
        """
        Saves the given data frames map to the S3 bucket as a zip archive.

        Args:
            dfs: A map of data frame names to the data frames to save.
            key_name: The name of the S3 object. If dfs contains more than on entry, the key name must end in '.zip'.
        """
        if len(dfs) > 1 and not self.__is_zip(key_name):
            raise ValueError(f'The key name {key_name} does not end in \'.zip\'. It needs to because all the data frames will be save to one zip archive.')

        buffer = BytesIO()
        if self.__is_zip(key_name):
            with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as zf:
                for df_name, df in dfs.items():
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer)
                    zf.writestr(df_name+'.csv', csv_buffer.getvalue())
        else:
            buffer = BytesIO()
            with GzipFile(None, 'wb', 9, buffer) as gz:
                gz.write(list(dfs.values())[0].to_csv().encode())

        self.s3.put_object(Body=buffer.getvalue(), Bucket=self.s3_bucket, Key=key_name, ContentEncoding='gzip')

    def save_file(self, source_file: str, key_name: str, content_encoding: str='') -> None:
        """
        Saves the given file to the S3 bucket.

        Args:
            source_file: The file to be uploaded.
            key_name: The name of the S3 object.
            content_encoding: The content encoding in S3. If this is set to gzip, the file will be compressed before upload.
        """
        with open(source_file, 'rb') as fp:
            if content_encoding == 'gzip':
                gz_buffer = BytesIO()
                with GzipFile(None, 'wb', 9, gz_buffer) as gz:
                    shutil.copyfileobj(fp, gz)

                self.s3.put_object(Body=gz_buffer.getvalue(), Bucket=self.s3_bucket, Key=key_name, ContentEncoding=content_encoding)
            else:
                self.s3.put_object(Body=fp.read(), Bucket=self.s3_bucket, Key=key_name, ContentEncoding=content_encoding)

    def save_dir(self, source_dir: str, key_name: str = '') -> None:
        """
        Saves the given directory to the S3 bucket either as a zip archive or as separate gzipped files.

        Args:
            source_dir: The directory that contains the files to save.
            key_name: The name of the S3 object. If the key name ends in '.zip', the directory will be uploaded as a zip archive.
                Otherwise, the files will be uploaded as separate gzip files and the key name will be treated as pre-fix for the uploaded files.
        """
        if self.__is_zip(key_name):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, mode='w', compression=ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        zf.write(f'{root}/{file}', arcname=file)

            self.s3.put_object(Body=zip_buffer.getvalue(), Bucket=self.s3_bucket, Key=key_name)
        else:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    self.save_file(f'{root}/{file}', f'{key_name}{file}', content_encoding='gzip')

    def load_df(self, key_name: str) -> DF:
        """
        Loads the data frame from the object with the given key name. If the name of the object ends in .zip,
        it will unzip the object on the fly use the first file the zip archive to load the data frame.

        Args:
            key_name: The name of the S3 object. If the key name ends in '.zip', it will extract the first file in the zip archive.

        Returns:
            The data frame.
        """

        obj = self.s3.get_object(Bucket=self.s3_bucket, Key=key_name)
        buffer = BytesIO(obj["Body"].read())

        if self.__is_zip(key_name):
            zf = ZipFile(buffer)

            if len(zf.namelist()) == 0:
                raise Exception(f'Could not load data frame because zip file {key_name} is empty.')

            buffer = BytesIO(zf.read(zf.namelist()[0]))

        return pd.read_csv(buffer)

    def load_dfs(self, key_name: str) -> Dict[str, DF]:
        """
        Loads the data frames with the given key name. If the name of the object ends in .zip,
        it will unzip the object on the fly load the data frames from the files in the zip archive.

        Args:
            key_name: The name of the S3 object. If the key name ends in '.zip', it will extract the first file in the zip archive.

        Returns:
            A map of data frame names to the data frames that have been loaded.
        """

        obj = self.s3.get_object(Bucket=self.s3_bucket, Key=key_name)
        buffer = BytesIO(obj["Body"].read())

        dfs = {}
        if self.__is_zip(key_name):
            zf = ZipFile(buffer)

            if len(zf.namelist()) == 0:
                raise Exception(f'Could not load data frame because zip file {key_name} is empty.')

            zf = ZipFile(buffer)
            for file_name in zf.namelist():
                dfs[file_name.replace('.csv', '')] = pd.read_csv(BytesIO(zf.read(file_name)))
        else:
            dfs[key_name.replace('.csv', '')] = pd.read_csv(buffer)

        return dfs
