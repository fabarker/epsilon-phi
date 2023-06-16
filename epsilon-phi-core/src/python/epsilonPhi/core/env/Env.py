import os
import configparser
from epsilonPhi.logging import logMessage as __, logger

APPDEV = 'APPDEV'
env = 'DEV'

DB_HOSTNAME = None
DB_USERNAME = None
DB_PASSWORD = None
DBL_PORT = None
DB_DRIVER = None
DB_DATABASE_NAME = None

DS_USERNAME = None
DS_PASSWORD = None

GSQ_CLIENT_ID = None
GSQ_CLIENT_SECRET = None
GSQ_GUID = None

class Env(object):
    _config_file = r'C:\Users\fabar\Repos\epsilon-phi\epsilon-phi-core\src\resources\config\env.ini'

    @staticmethod
    def is_dev():
        return True if 'DEV' in os.environ.get(APPDEV, "").upper() else False

    @staticmethod
    def is_prod():
        return True if 'PROD' in os.environ.get(APPDEV, "").upper() else False

    @staticmethod
    def is_uat():
        return True if 'UAT' in os.environ.get(APPDEV, "").upper() else False

    @staticmethod
    def get_env():
        return os.environ.get(APPDEV, env).upper()

    @staticmethod
    def set_env(envName: str):
        os.environ[APPDEV] = envName
        Env._initalize()

    @staticmethod
    def _reinitalize(envName: str):
        Env.set_env(envName)

    @staticmethod
    def read_config():
        cp = configparser.ConfigParser()
        cp.read(Env._config_file)
        return cp

    @staticmethod
    def _initalize():

        global DB_HOSTNAME
        global DB_USERNAME
        global DB_PASSWORD
        global DB_PORT
        global DB_DRIVER
        global DB_DATABASE_NAME

        global DS_USERNAME
        global DS_PASSWORD

        global GSQ_CLIENT_ID
        global GSQ_CLIENT_SECRET
        global GSQ_GUID

        currEnv = Env.get_env()
        _config = Env.read_config()

        DB_HOSTNAME = _config.get(currEnv, 'HOSTNAME')
        DB_USERNAME = _config.get(currEnv, 'USERNAME')
        DB_PASSWORD = _config.get(currEnv, 'PASSWORD')
        DB_PORT = _config.get(currEnv, 'PORT')
        DB_DRIVER = _config.get(currEnv, 'DRIVER')
        DB_DATABASE_NAME = _config.get(currEnv, 'DATABASE_NAME')

        DS_USERNAME = _config.get('DATASTREAM', 'USERNAME')
        DS_PASSWORD = _config.get('DATASTREAM', 'PASSWORD')

        GSQ_CLIENT_ID = _config.get('GSQUANT', 'CLIENT_ID')
        GSQ_CLIENT_SECRET = _config.get('GSQUANT', 'CLIENT_SECRET')
        GSQ_GUID = _config.get('GSQUANT', 'GUID')

        logger.info('Intialized Environemnt with {}'.format(currEnv))

        #from epsilonPhi.dataModel.dataSources.GlobalDataSource import GlobalDataSource
        #GlobalDataSource.initialize()

Env._initalize()


