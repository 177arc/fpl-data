import logging
import sys
import os
import unittest

class TestComp(unittest.TestCase):

    def test_data_all(self):
        logging.info(f'Setting the ENV environment variable to Comp-Test to run the notebook code in component testing mode ...')
        os.environ['ENV'] = 'Comp-Test'

        logging.info(f'Executing component test ...')
        # Execute script generated from notebook
        if 'prep_data' in sys.modules: del sys.modules['prep_data']
        import prep_data

        logging.info(f'Component test executed successfully.')