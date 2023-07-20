from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries

class CFactor(CTimeSeries):
    def __init__(self, df=None, factor_name=None, schema=None):
        super(CFactor, self).__init__(df)

    def get_factor_time_series(self):
        return self

