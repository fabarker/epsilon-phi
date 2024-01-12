

class PositionResolver(object):

    def __init__(self,
                 signal_df,
                 position_methodology):

        self._optimzer = None
        self._covariance = None

    @property
    def optimzer(self):
        return self.optimzer

    def set_assets(self, assetList):
        pass

    def set_covariance(self, covariance):
        pass

    def set_optimizer(self, optimizer):
        pass

    def get_constraints(self):
        pass

    def get_positions(self, rebalancing_date):
        pass
