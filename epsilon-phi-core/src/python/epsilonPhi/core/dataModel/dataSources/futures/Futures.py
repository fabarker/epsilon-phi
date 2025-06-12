import numpy as np
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.timeSeries.timeSeriesMain import *
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.ep_strategies.utils.StrategyUtils import StrategyUtils
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from sqlalchemy import desc
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

    def get_field_value_for_ticker(self, ticker, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)

        return session.query(field).filter(FutureSpec.ticker == ticker).scalar()

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

    def get_security_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'security')

    def get_ticker_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'ticker')

    def get_available_forward_positions_from_instrument_menomic(self, mnemonic):
        return np.unique(self.get_field_values_for_mnemonic(mnemonic, 'position_forward')).astype(int)

    def get_continuous_series_single_ticker(self, ticker):
        uid = self.get_field_value_for_ticker(ticker, 'uid')
        return self.get_continuous_series_single_uid(uid)

    def get_continuous_series_single_uid(self, uid):
        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource as gds

        res = gds().get_time_series_data_from_uid(uid, ts_type=TimeSeriesType.LEVELS, returnsType=ReturnsType.DIFFERENCE)
        res.set_attribute_single('forward', self.get_series_position_forward_from_uid(uid))
        res.set_attribute_single('mnemonic', self.get_futures_mnemonic_from_uid(uid))
        res.set_attribute_single('ticker', self.get_ticker_from_uid(uid))
        res.set_attribute_single('name', self.get_name_from_uid(uid))
        res.set_attribute_single('security', self.get_security_from_uid(uid))
        res.columns = res.columns.swaplevel('mnemonic', res.columns.get_level_values(0).name)
        return res.deepcopy()

    def _resolve_ticker_list(self, list):
        return sorted(list, key=lambda x: x.replace('.', '~'))

    def get_futures_continuous_series_forward(self, mnemonic, forward):

        # query the database for the information we need
        res = session.query(FutureSpec.ticker)\
                       .filter(FutureSpec.future == mnemonic,
                               FutureSpec.position_forward == int(forward))\
                       .order_by(desc(FutureSpec.ticker))\
                       .all()

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.DIFFERENCE)
        for ticker in res:
            tmp = self.get_continuous_series_single_ticker(ticker[0])
            tmp.drop_attributes(['ticker', 'uid', 'name'])
            ts = ts.backfill_levels(tmp)
        return ts.deepcopy()


    def get_futures_continuous_series(self, mnemonic):

        # Get the position forward for the mnemonic
        fwds = self.get_available_forward_positions_from_instrument_menomic(mnemonic)

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS)
        for fwd in fwds:
            ts = ts.concat(self.get_futures_continuous_series_forward(mnemonic, fwd))
        return ts.deepcopy()

    def get_all_continuous_series(self, mnemonic):

        res = session.query(FutureSpec.ticker) \
            .filter(FutureSpec.future == mnemonic) \
            .order_by(desc(FutureSpec.start_date)) \
            .all()

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.DIFFERENCE)
        for ticker in res:
            tmp = self.get_continuous_series_single_ticker(ticker[0])
            ts = ts.concat(tmp)
        return ts.deepcopy()


    def get_front_futures_continuous_series_settlement_price(self, mnemonic):
        return (self.get_futures_continuous_series_settlement_price(mnemonic).
                select_subset_attribute('forward', 0))

    def get_back_futures_continuous_series_settlement_price(self, mnemonic):
        return (self.get_futures_continuous_series_settlement_price(mnemonic).
                select_subset_attribute('forward', 1))

    def get_futures_continuous_series_settlement_price(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'PS')

    def get_futures_continuous_series_open_interest(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'OI')

    def get_futures_continuous_volume(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'VM')

    def get_OHLC_for_continuous_future(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field',
                                                                        ['PO','PH','PL','PS'])

    def get_futures_carry_continuous(self, mnemonic):
        front = self.get_front_futures_continuous_series_settlement_price(mnemonic)
        back = self.get_back_futures_continuous_series_settlement_price(mnemonic)
        return np.log(back.division_over_common_dates(front))

    def estimate_bid_ask_prices_for_futures_continuous(self, mnemonic):
        ohlc = self.get_OHLC_for_continuous_future(mnemonic).dropna(how='any', axis=0)
        bid, ask = StrategyUtils.estimate_bid_ask_prices(
            open=ohlc.select_subset_attribute('field', 'PO'),
            high=ohlc.select_subset_attribute('field', 'PH'),
            low=ohlc.select_subset_attribute('field', 'PL'),
            close=ohlc.select_subset_attribute('field', 'PS')
        )

        prices = bid.concat(ask)
        prices.set_attribute_single('price_quote_type', ['bid', 'ask'])
        return prices.copy()

    def display_futures_info(self):
        sessionMgr.show_table(FutureSpec)


if __name__ == "__main__":


    self = Futures()
    df_ = CTimeSeries(ts_type=TimeSeriesType.LEVELS)

    codes = [
        "NHO",
        "NRB",
        "NCL",
        "NNG",
        "NHG",
        "NGC",
        "NSL",
        "CFD",
        "CLG",
        "CLD",
        "CNR",
        "CCF",
        "CZO",
        "CWF",
        "NSB",
        "CSY",
        "CSN",
        "NCC",
        "NKC",
        "NJO",
        "NCT",
        "CMS"
    ]

    frame_dict = {}
    for name in codes:
        res = self.get_futures_continuous_series_settlement_price(name).dropna()
        frame_dict[name] = res.copy()

    from epsilonPhi.core.utils.ExcelUtils import *
    ExcelUtils.dict_to_excel(
        frame_dict,
        filename='futures.xlsx',
        include_index=True,
    )


