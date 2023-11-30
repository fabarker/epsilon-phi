from epsilonPhi.core.env.Env import Env
from epsilonPhi.core.lib.Constants import InfinityTime

class CAppConfig(object):

    _setup = False
    _estimationMgr = None
    _configUtil = None
    _asOfDate = InfinityTime
    _BaseModel = None

    @staticmethod
    def reset():
        CAppConfig._setup = False
        CAppConfig._configUtil = False
        CAppConfig._estimationMgr = False

    @staticmethod
    def setup(app='DEV'):

        if app != Env.get_env():
            Env._reinitalize(app)

        from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
        CAppConfig._estimationMgr = EstimationMgr()

        from epsilonPhi.core.config.configUtil import CConfigUtil
        CAppConfig._configUtil = CConfigUtil()

        from epsilonPhi.core.modelFactory.modelFactory import BaseModel
        CAppConfig._BaseModel = BaseModel.setup_default_model()

        CAppConfig._setup = True

    @staticmethod
    def get_BaseModel():
        CAppConfig.check_for_setup()
        return CAppConfig._BaseModel

    @staticmethod
    def check_for_setup():
        if CAppConfig._setup is False:
            CAppConfig.setup(Env.get_env())
        return True
    @staticmethod
    def get_estimation_mgr():
        CAppConfig.check_for_setup()
        return CAppConfig._estimationMgr

    @staticmethod
    def get_as_of_date():
        return CAppConfig._asOfDate

    @staticmethod
    def set_as_of_date(as_of_date):
        CAppConfig._asOfDate = as_of_date

    @staticmethod
    def get_config_util():
        CAppConfig.check_for_setup()
        return CAppConfig._configUtil






