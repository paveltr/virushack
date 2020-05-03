import logging
from lib.get_data_from_sql import data_pipeline
from lib.create_features import features_pipeline
from lib.get_predictions import prediction_pipeline
from lib.encoders import TargetEncoderCV
import os
from predict import PredictionManager

# only for local tests
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# os.environ['BQ_PROJECT'] = r'pp-import-production'
# os.environ['BQ_JSON'] = r'config/bq.json'
# os.environ['MYSQL_SSH'] = r'config/mysql-workbench'
# os.environ['MYSQL_CONFIG'] = r'config/config_prod.yaml'
# os.environ['STORAGE_JSON'] = r'config/storage.json'
# os.environ['LACE_BUCKET'] = r'pp-lace-score-engine-production'
# os.environ['LACE_PROJECT'] = r'pp-suspect-production'

# create logger with 'spam_application'
logger = logging.getLogger('create_data')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('tmp/create_data.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s[%(name)s] - [%(levelname)s]: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def send_to_the_db():
    PredictionManager.save_to_db(r'data/predictions.csv')


def pipeline():
    """API Call
    Create pandas dataframe with latest actual data for predictions
    """

    logger.info('Started pulling data')
    data_pipeline()
    logger.info('Started creating features')
    features_pipeline()
    logger.info('Start predicting')
    prediction_pipeline()
    logger.info('Sending data to the MySQL')
    send_to_the_db()
    logger.info('Done')


if __name__ == '__main__':
    pipeline()
