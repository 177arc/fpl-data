import os
import zipfile
import logging as log
import boto3
import re
from shell_utils import shell
import subprocess

NOTEBOOK_PATH = 'prep_data.ipynb'
DIST_PATH = 'dist'
DEPLOY_ZIP_PATH = f'{DIST_PATH}/fpl-data.zip'
S3_BUCKET = 'lambdas.177arc.net'
S3_PATH = 'deploy'
LAMBDA_NAME = 'fpl-data'

log.basicConfig(level=log.INFO, format='%(message)s')


class WorkingDirSwitcher(object):
    def __init__(self, working_dir):
        self.working_dir = working_dir

    def __enter__(self):
        self.prev_working_dir = os.getcwd()
        os.chdir(self.working_dir)

    def __exit__(self, type, value, traceback):
        os.chdir(self.prev_working_dir)


def __execute(command: str, working_dir: str = '.'):
    with WorkingDirSwitcher(working_dir):
        result = shell(command, capture=True, silent=True, check=False)
        log.error(result.stderr)
        log.info(result.stdout)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(cmd=command, returncode=result.returncode)


def zip_dir(path: str, zip_file: str, incl_regex: str = '.*') -> None:
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = f'{root}/{file}'
                if re.match(incl_regex, file_path):
                    zf.write(file_path)


def unit_test():
    log.info('Running unit tests ...')
    __execute('python -m pytest tests/unit/')
    log.info('Unit tests completed.')


def int_test():
    log.info('Running integration tests ...')
    __execute('python -m unittest discover -s "."', './tests/integration')
    log.info('Integration tests completed.')


def e2e_test():
    log.info('Running integration tests ...')
    __execute('python -m unittest discover -s "."', './tests/e2e')
    log.info('Integration tests completed.')


def build():
    log.info(f'Exporting Jupyter notebook ...')
    __execute('jupyter nbconvert --to script prep_data.ipynb')
    log.info(f'Jupyter notebook exported.')


def deploy(function_name: str = 'fpl-data-test'):
    os.makedirs(DIST_PATH, exist_ok=True)

    if os.path.isfile(DEPLOY_ZIP_PATH):
        log.info(f'Removing deployment package {DEPLOY_ZIP_PATH} ...')
        os.remove(DEPLOY_ZIP_PATH)

    log.info(f'Creating deployment package {DEPLOY_ZIP_PATH} ...')
    zip_dir('.', DEPLOY_ZIP_PATH, r'^\.[\/\\]([^\/\\]*|20.*)(\.py|\.ipynb|\.csv)$')

    session = boto3.Session(profile_name='fpl-data-ci', region_name='eu-west-2')
    lmda = session.client('lambda')
    log.info(f'Deploying file {DEPLOY_ZIP_PATH} to AWS lambda ...')
    with open(DEPLOY_ZIP_PATH, mode='rb') as dzf:
        _ = lmda.update_function_code(
            FunctionName=function_name,
            Publish=True,
            ZipFile=dzf.read()
        )

    log.info('Deployment to AWS done.')


if __name__ == "__main__":
    deploy()
