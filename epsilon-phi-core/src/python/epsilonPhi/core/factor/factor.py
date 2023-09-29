from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice

class CFactor(CSlice):
    def __init__(self, df=None, factor_name=None, schema=None):
        super(CFactor, self).__init__(df)

    def get_factor_time_series(self):
        return self

