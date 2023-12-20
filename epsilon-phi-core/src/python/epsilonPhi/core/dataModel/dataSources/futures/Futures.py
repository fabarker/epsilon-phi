import numpy as np
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.timeSeries.timeSeriesMain import *
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

@SingletonDecorator
class Futures(object):

    _cache = pd.DataFrame()
    _columns = ([column.name for column in FutureSpec.__table__.columns] +
                [column.name for column in TimeSeriesSpec.__table__.columns])


    def __init__(self):
        pass

    def get_field_value_for_uid(self, uid, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)

        return session.query(field).filter(FutureSpec.uid == uid).scalar()

    def get_field_values_for_mnemonic(self, mnemonic, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)
        res = session.query(field).filter(FutureSpec.future == mnemonic).all()

        if len(res) > 0:
           return [x[0] for x in res]
        else:
           return None

    def get_uids_from_instrument_mnemonic(self, mnemonic):
        return self.get_field_values_for_mnemonic(mnemonic, 'uid')

    def get_series_position_forward_from_uid(self, uid: int):
        return int(self.get_field_value_for_uid(uid, 'position_forward'))

    def get_tick_size_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'tick_size')

    def get_tick_value_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'tick_value')

    def get_contract_size_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'contract_size')

    def get_denominated_currency_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'denominated_currency')

    def get_security_type_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'security_type')

    def get_futures_mnemonic_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'future')

    def get_name_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'name')

    def get_ticker_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'ticker')

    def get_continuous_series_single_uid(self, uid):
        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource as gds

        res = gds().get_time_series_data_from_uid(uid, ts_type=TimeSeriesType.LEVELS)
        res.set_attribute_single('forward', self.get_series_position_forward_from_uid(uid))
        res.set_attribute_single('mnemonic', self.get_futures_mnemonic_from_uid(uid))
        res.set_attribute_single('ticker', self.get_ticker_from_uid(uid))
        res.set_attribute_single('name', self.get_name_from_uid(uid))
        res.columns = res.columns.swaplevel('mnemonic', res.columns.get_level_values(0).name)
        return res.deepcopy()

    def get_futures_continuous_series(self, mnemonic):

        # Get all uids corresponding to this mnemonic,
        # there may be more than 1 due to later contracts
        uids = self.get_uids_from_instrument_mnemonic(mnemonic)

        # If we have uids, then fetch them
        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS)
        for uid in uids:
            ts = ts.concat(self.get_continuous_series_single_uid(uid))
        return ts.deepcopy()

    def get_futures_continuous_series_settlement_price(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'PS')

    def get_futures_continuous_series_open_interest(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'OI')

    def get_futures_continuous_volume(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'VM')

    def get_futures_instrument_liquidity(self):
        pass

    def get_futures_instrument_price_impact(self):
        pass






if __name__ == "__main__":


    self = Futures()
    mnemonic = 'OMN'

    ts = self.get_futures_continuous_series_settlement_price(mnemonic)




