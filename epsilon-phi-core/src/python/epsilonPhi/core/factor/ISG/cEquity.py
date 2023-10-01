from epsilonPhi.core.factor.cAbstractFactor import AbstractFactor
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource as gds

FACTOR_DEFINITION = [('MSWRLDL', 0.5), ('MSWRLD$', 0.5)]
FREQUENCY = Frequency.MONTHLY

class EquityFactor(AbstractFactor):
    def __init__(self):
        super(EquityFactor, self).__init__(factorName='ISGEquity')


    def construct_factor_time_series(self):
        pass


