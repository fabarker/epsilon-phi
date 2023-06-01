from epsilonPhi.core.env.Env import Env

class CAppConfig(object):
    _setuo = False

    @staticmethod
    def setup(app='DEV'):
        Env._reinitalize(app)
        _setup = True

