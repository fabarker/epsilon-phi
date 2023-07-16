from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.schema.Schema import CContext
from datetime import datetime
import pandas as pd
import numpy as np
import math

class CAsset(CTimeSeries, CAssetInf):
    def __init__(self,
                 ticker: str,
                 currency: str,
                 hedging_ratio: float,
                 schema: CContext,
                 dataframe=None):

        super(CAsset, self).__init__(dataframe)
        self._currency = currency
        self._ticker = ticker
        self._hedging_ratio = hedging_ratio
        self._schema = schema

    @property
    def asset_ticker(self):
        return self._ticker
    @property
    def currency(self):
        return self._currency
