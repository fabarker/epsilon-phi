from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, TypeDecorator, text, Boolean, LargeBinary
from typing import Any
from epsilonPhi.core.lib.Decorators import auto_repr
from sqlalchemy.orm import relationship, declarative_base, declared_attr
from epsilonPhi.core.dataModel.alchemist.BaseData import Base
from epsilonPhi.core.utils.DateUtils import DateUtils
from decimal import Decimal
import numpy as np
import pandas as pd
import math


class FloatOrNone(TypeDecorator):
    impl = Float

    def process_bind_param(self, value, dialect) -> Any:
        if isinstance(value, (float, Decimal, np.float64, np.float32)) and math.isnan(float(value)):
                return None
        return value

class CategoryTableMapping(Base):
    __tablename__ = 'category_table_mapping'

    category = Column(String(50), nullable=True, primary_key=True)
    spec_table_name = Column(String(50), nullable=True)
    table_name = Column(String(50), nullable=True)
    spec_table_ORM = Column(String(50), nullable=True)
    table_ORM = Column(String(50), nullable=True)

@auto_repr
class TimeSeriesSpec(Base):
    __tablename__ = 'time_series_spec'

    uid = Column(Integer, primary_key=True, index=True)
    provider = Column(String(250), nullable=True)
    ticker = Column(String(50), nullable=True)
    region = Column(String(100), nullable=True)
    name = Column(String(255), nullable=True)
    category = Column(String(50), ForeignKey('category_table_mapping.category'), nullable=False)
    datasource = Column(String(50), nullable=True)
    symbol = Column(String(50), nullable=True)
    frequency = Column(String(10), nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'time_series_spec'}
    __map = relationship("CategoryTableMapping", foreign_keys=[category], primaryjoin='TimeSeriesSpec.category == CategoryTableMapping.category')

    @property
    def table_name(self):
        if hasattr(self, '_TimeSeriesSpec__map.table_name'):
            return self._TimeSeriesSpec__map.table_name
        else:
            return None


@auto_repr
class TimeSeries(Base):
    __abstract__ = True

    @declared_attr
    def uid(cls):
        return Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

    @declared_attr
    def date(cls):
        return Column(DateTime, primary_key=True)

    __mapper_args__ = {'polymorphic_identity': 'time_series'}


@auto_repr
class BondIndexSpec(TimeSeriesSpec):
    __tablename__ = 'bond_index_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    pricing_currency = Column(String(3), nullable=False, index=True)

    sector = Column(String(20), nullable=False)
    rating = Column(String(10), nullable=False)
    maturity_band = Column(String(20), nullable=False)
    maturity = Column(Integer, nullable=True)

    @property
    def denominated_currency(self):
        return self.pricing_currency

    @property
    def exposure_currency(self):
        return self.pricing_currency

    @property
    def hedge_ratio(self):
        return 0

    __mapper_args__ = {'polymorphic_identity': 'bond_index_spec'}

@auto_repr
class BondIndex(TimeSeries):
    __tablename__ = 'bond_index'

    uid = Column(Integer, ForeignKey('bond_index_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    DM = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    RY = Column(FloatOrNone, nullable=True)
    CX = Column(FloatOrNone, nullable=True)
    IN = Column(FloatOrNone, nullable=True)
    YTW = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'bond_index'}
    _spec = relationship("BondIndexSpec", foreign_keys=[uid])


############### Equity Indicies ##############
@auto_repr
class EquityIndexSpec(TimeSeriesSpec):
    __tablename__ = 'equity_index_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

    # Map DB column "denominated_currency" → Python attribute "_denominated_currency"
    #_denominated_currency = Column(
    #    "denominated_currency",
    #    String(3),
    #    key="_denominated_currency",
    #    nullable=False,
    #    index=True
    #)

    #_exposure_currency = Column(
    #    "exposure_currency",
    #    String(3),
    #    key="_exposure_currency",
   #     nullable=False,
   #     index=True
    #)

    denominated_currency = Column(String(3), nullable=False, index=True)
    exposure_currency = Column(String(3), nullable=False, index=True)
    hedge_ratio = Column(FloatOrNone, nullable=True)

    # UID → (denominated_currency, exposure_currency)
    currency_overrides = {
        18140: ('USD', 'AAP'),
        18141: ('AAP', 'AAP'),
        18144: ('USD', 'ACW'),
        18145: ('ACW', 'ACW'),
        18146: ('USD', 'AEM'),
        18147: ('AEM', 'AEM'),
        18166: ('USD', 'ACA'),
        18167: ('ACA', 'ACA'),
        18179: ('USD', 'EFE'),
        18180: ('EFE', 'EFE'),
        18183: ('USD', 'EEP'),
        18184: ('EEP', 'EEP'),
        18185: ('USD', 'ELA'),
        18186: ('ELA', 'ELA'),
        18191: ('USD', 'EME'),
        18192: ('EME', 'EME'),
        18193: ('USD', 'EMA'),
        18194: ('EMA', 'EMA'),
        18195: ('USD', 'EMM'),
        18196: ('EMM', 'EMM'),
        18199: ('USD', 'ERP'),
        18200: ('ERP', 'ERP'),
        18203: ('USD', 'ERP'),
        18204: ('EXU', 'EXU'),
        18265: ('USD', 'PAC'),
        18266: ('PAC', 'PAC'),
        18277: ('USD', 'PXJ'),
        18278: ('PXJ', 'PXJ'),
        18322: ('USD', 'WLD'),
        18323: ('WLD', 'WLD'),
    }

    __mapper_args__ = {'polymorphic_identity': 'equity_index_spec'}

@auto_repr
class EquityIndex(TimeSeries):
    __tablename__ = 'equity_index'

    uid = Column(Integer, ForeignKey('equity_index_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    DY = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    PI = Column(FloatOrNone, nullable=True)
    MV = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'equity_index'}
    _spec = relationship("EquityIndexSpec", foreign_keys=[uid])

############### Commodity Indicies ##############
@auto_repr
class CommodityIndexSpec(TimeSeriesSpec):
    __tablename__ = 'commodity_index_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    denominated_currency = Column(String(3), nullable=False, index=True)
    exposure_currency = Column(String(3), nullable=False, index=True)
    hedge_ratio = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'commodity_index_spec'}

@auto_repr
class CommodityIndex(TimeSeries):
    __tablename__ = 'commodity_index'

    uid = Column(Integer, ForeignKey('commodity_index_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)
    X = Column(FloatOrNone, nullable=True)

    @property
    def _X(self):
        return 'RI'

    __mapper_args__ = {'polymorphic_identity': 'commodity_index'}
    _spec = relationship("CommodityIndexSpec", foreign_keys=[uid])

############### FX Rates ##############

@auto_repr
class FXRateSpec(TimeSeriesSpec):
    __tablename__ = 'fx_rates_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    foreign_currency = Column(String(3), nullable=False, index=True)
    domestic_currency = Column(String(3), nullable=False, index=True)
    bbid = Column(String(6), nullable=False)
    maturity = Column(String(10), nullable=False)
    type = Column(String(1), nullable=False)

    __mapper_args__ = {'polymorphic_identity': 'fx_rate_spec'}


@auto_repr
class FXRate(TimeSeries):
    __tablename__ = 'fx_rates'

    uid = Column(Integer, ForeignKey('fx_rates_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    pricing_location = Column(String(6), primary_key=True)
    EB = Column(FloatOrNone, nullable=True)
    ER = Column(FloatOrNone, nullable=True)
    EO = Column(FloatOrNone, nullable=True)
    X = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'fx_rates'}
    _spec = relationship("FXRateSpec", foreign_keys=[uid])

    @property
    def bbid(self):
        return self._spec.bbid

    @property
    def foreign_currency(self):
        return self._spec.foreign_currency

    @property
    def domestic_currency(self):
        return self._spec.domestic_currency

    @property
    def base_currency(self):
        return self.foreign_currency

    @property
    def counter_currency(self):
        return self.domestic_currency

    @property
    def maturity(self):
        return self._spec.maturity

    @property
    def _X(self):
        return ''

############### Yield Curves ##############

@auto_repr
class YieldCurveSpec(TimeSeriesSpec):
       __tablename__ = 'yield_curve_spec'

       uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
       currency = Column(String(3), nullable=False, index=True)
       maturity = Column(String(10), nullable=False)
       type = Column(String(10), nullable=False)

       __mapper_args__ = {'polymorphic_identity': 'yield_curve_spec'}

       @property
       def denominated_currency(self):
           return self.currency
       @property
       def exposure_currency(self):
           return self.currency

@auto_repr
class YieldCurve(TimeSeries):
    __tablename__ = 'yield_curve'

    uid = Column(Integer, ForeignKey('yield_curve_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)
    RY = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'yield_curve'}
    _spec = relationship("YieldCurveSpec", foreign_keys=[uid])

    @property
    def currency(self):
        return self._spec.currency

    @property
    def maturity(self):
        return self._spec.maturity

    @property
    def type(self):
        return self._spec.type

############### Interest Rates ##############

@auto_repr
class InterestRateSpec(TimeSeriesSpec):
       __tablename__ = 'interest_rate_spec'

       uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
       currency = Column(String(3), nullable=False, index=True)
       maturity = Column(String(10), nullable=False)
       type = Column(String(10), nullable=False)

       __mapper_args__ = {'polymorphic_identity': 'interest_rate_spec'}

       @property
       def denominated_currency(self):
           return self.currency

       @property
       def exposure_currency(self):
           return self.currency


@auto_repr
class InterestRate(TimeSeries):
    __tablename__ = 'interest_rate'

    uid = Column(Integer, ForeignKey('interest_rate_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    IB = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    IR = Column(FloatOrNone, nullable=True)
    IO = Column(FloatOrNone, nullable=True)
    X = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'interest_rate'}
    _spec = relationship("InterestRateSpec", foreign_keys=[uid])

    @property
    def maturity(self):
        return self._spec.maturity

    @property
    def type(self):
        return self._spec.type

    @property
    def _X(self):
        return 'IR'


############### Hedge Funds ##############
@auto_repr
class HedgeFundIndexSpec(TimeSeriesSpec):
       __tablename__ = 'hedge_fund_index_spec'

       uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

       denominated_currency = Column(String(3), nullable=False, index=True)
       exposure_currency = Column(String(3), nullable=False, index=True)
       hedge_ratio = Column(Integer, nullable=True)
       strategy_type = Column(String(150), nullable=False, index=True)

       __mapper_args__ = {'polymorphic_identity': 'hedge_fund_spec'}
@auto_repr
class HedgeFundIndex(TimeSeries):
    __tablename__ = 'hedge_fund_index'

    uid = Column(Integer, ForeignKey('hedge_fund_index_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)
    RI = Column('X', FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'hedge_fund_index'}
    _spec = relationship("HedgeFundIndexSpec", foreign_keys=[uid])

    @property
    def hedge_ratio(self):
        return self._spec.hedge_ratio

    @property
    def _X(self):
        return 'RI'


# ############### Implied Volatility ################

@auto_repr
class ImpliedVolatilitySpec(Base):
    __tablename__ = 'implied_volatility_spec'

    security = Column(String(100), primary_key=True, nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    region = Column(String(100), nullable=False, index=True)
    currency = Column(String(3), nullable=False, index=True)
    name = Column(String(100), nullable=False, index=True)
    ticker = Column(String(100), nullable=False, index=True)

@auto_repr
class ImpliedVolatilityNew(TimeSeries):
    __tablename__ = 'implied_volatility_new'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    date = Column(DateTime, primary_key=True)

    pricing_location = Column(String(6), primary_key=True)
    strike_reference = Column(String(20), nullable=True, index=True)
    relative_strike = Column(String(20), nullable=True, index=True)
    tenor = Column(String(9), nullable=False, index=True)

    bid = Column(FloatOrNone, nullable=True)
    mid = Column(FloatOrNone, nullable=True)
    ask = Column(FloatOrNone, nullable=True)

    security = Column(String(100), nullable=False, index=True)

    __mapper_args__ = {'polymorphic_identity': 'implied_volatility_new'}
    _spec = relationship("TimeSeriesSpec", foreign_keys=[uid])

    @property
    def expiry(self):
        return DateUtils.Rdate_to_mat(self.tenor)

@auto_repr
class ImpliedVolatility(TimeSeries):
    __tablename__ = 'implied_volatility'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    date = Column(DateTime, primary_key=True)

    pricing_location = Column(String(6), primary_key=True)
    strike_reference = Column(String(20), nullable=True)
    relative_strike = Column(String(20), nullable=True)
    tenor = Column(String(9), nullable=False, index=True)

    bid = Column(FloatOrNone, nullable=True)
    mid = Column(FloatOrNone, nullable=True)
    ask = Column(FloatOrNone, nullable=True)

    security = Column(String(100), nullable=False, index=True)

    __mapper_args__ = {'polymorphic_identity': 'implied_volatility'}
    _spec = relationship("TimeSeriesSpec", foreign_keys=[uid])

    @property
    def expiry(self):
        return DateUtils.Rdate_to_mat(self.tenor)

    #@property
    #def expiration_date(self):
    #    return self.date + relativedelta(years=self.expiry)

    @staticmethod
    def getDataframe(**kwargs):

        from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
        res = SessionMgr().getSessionFactory().query(ImpliedVolatility.date,
                            ImpliedVolatility.bid,
                            ImpliedVolatility.mid,
                            ImpliedVolatility.ask).filter_by(**kwargs).all()
        df = pd.DataFrame(res)
        df.columns = ['date','bid','mid','ask']
        return df.set_index('date', drop=True).sort_index()

@auto_repr
class EquitySpec(TimeSeriesSpec):
      __tablename__ = 'equity_spec'

      uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

      ISIN = Column(String(100), nullable=False, index=True)
      SEDOL = Column(String(100), nullable=False, index=True)

      exchange = Column(String(100), nullable=False, index=True)
      exchange_code = Column(String(20), nullable=False, index=True)
      exchange_mnemonic = Column(String(20), nullable=False, index=True)

      sector = Column(String(255), nullable=False, index=True)
      industry_group = Column(String(255), nullable=False, index=True)
      industry = Column(String(255), nullable=False, index=True)
      sub_industry = Column(String(255), nullable=False, index=True)

      denominated_currency = Column(String(3), nullable=False, index=True)
      exposure_currency = Column(String(3), nullable=False, index=True)
      hedge_ratio = Column(FloatOrNone, nullable=True)

      __mapper_args__ = {'polymorphic_identity': 'equity_spec'}

@auto_repr
class Equity(TimeSeries):
    __tablename__ = 'equity'

    uid = Column(Integer, ForeignKey('equity_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    PI = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    DY = Column(FloatOrNone, nullable=True)
    MV = Column(FloatOrNone, nullable=True)

    PE = Column(FloatOrNone, nullable=True)
    EPS = Column(FloatOrNone, nullable=True)
    PTBV = Column(FloatOrNone, nullable=True)
    VO = Column(FloatOrNone, nullable=True)
    EPS_Est_12M = Column('EPS1FD12', FloatOrNone, nullable=True)
    NOSH = Column(FloatOrNone, nullable=True)
    SALES = Column('WC01001', FloatOrNone, nullable=True)
    APC = Column(FloatOrNone, nullable=True)
    EY_Est_12M = Column('529E', FloatOrNone, nullable=True)
    EPS_Est_12M_DS = Column('DIEP', FloatOrNone, nullable=True)
    PE_Est_12M = Column('DIPE', FloatOrNone, nullable=True)

    ASK = Column('PA', FloatOrNone, nullable=True)
    BID = Column('PB', FloatOrNone, nullable=True)
    LOW = Column('PL', FloatOrNone, nullable=True)
    HIGH = Column('PH', FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'equity'}
    _spec = relationship("EquitySpec", foreign_keys=[uid])

@auto_repr
class ETFSpec(TimeSeriesSpec):
      __tablename__ = 'etf_spec'

      uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

      ISIN = Column(String(100), nullable=False, index=True)
      SEDOL = Column(String(100), nullable=False, index=True)

      exchange = Column(String(100), nullable=False, index=True)
      exchange_code = Column(String(20), nullable=False, index=True)
      exchange_mnemonic = Column(String(20), nullable=False, index=True)

      denominated_currency = Column(String(3), nullable=False, index=True)
      exposure_currency = Column(String(3), nullable=False, index=True)
      hedge_ratio = Column(FloatOrNone, nullable=True)

      __mapper_args__ = {'polymorphic_identity': 'etf_spec'}

@auto_repr
class ETF(TimeSeries):
    __tablename__ = 'etf'

    uid = Column(Integer, ForeignKey('etf_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    PI = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    DY = Column(FloatOrNone, nullable=True)
    MV = Column(FloatOrNone, nullable=True)

    VO = Column(FloatOrNone, nullable=True)
    NOSH = Column(FloatOrNone, nullable=True)

    ASK = Column('PA', FloatOrNone, nullable=True)
    BID = Column('PB', FloatOrNone, nullable=True)
    LOW = Column('PL', FloatOrNone, nullable=True)
    HIGH = Column('PH', FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'etf'}
    _spec = relationship("ETFSpec", foreign_keys=[uid])


@auto_repr
class FutureSpec(TimeSeriesSpec):
      __tablename__ = 'future_spec'

      uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

      future = Column(String(3), nullable=True, index=True)
      start_date = Column(DateTime, nullable=True)
      tick_size = Column(FloatOrNone,  index=True)
      tick_value = Column(FloatOrNone,  index=True)
      contract_size = Column(FloatOrNone,  index=True, nullable=True, default=None)
      cycle = Column(String(100))

      exchange = Column(String(100), nullable=False, index=True)
      exchange_name = Column(String(100), nullable=False, index=True)

      security = Column(String(100), nullable=False, index=True)
      security_name = Column(String(100), nullable=False, index=True)
      security_type = Column(String(100), nullable=False, index=True)
      security_unit = Column(String(100), nullable=False, index=True)

      denominated_currency = Column(String(3), nullable=False, index=True)
      exposure_currency = Column(String(3), nullable=False, index=True)
      hedge_ratio = Column(FloatOrNone, nullable=True)
      position_forward = Column(Integer, nullable=False, default=0)

      __mapper_args__ = {'polymorphic_identity': 'future_spec'}

@auto_repr
class Future(TimeSeries):
    __tablename__ = 'future'

    uid = Column(Integer, ForeignKey('future_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    L = Column(FloatOrNone, nullable=True)
    OI = Column(FloatOrNone, nullable=True)
    PH = Column(FloatOrNone, nullable=True)
    PL = Column(FloatOrNone, nullable=True)
    PO = Column(FloatOrNone, nullable=True)
    PS = Column(FloatOrNone, nullable=True)
    VM = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'future'}
    _spec = relationship("FutureSpec", foreign_keys=[uid])

@auto_repr
class FactorSpec(TimeSeriesSpec):
    __tablename__ = 'factor_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    factor = Column(String(100))
    universe = Column(String(100))

@auto_repr
class Factor(TimeSeries):
    __tablename__ = 'factor'

    uid = Column(Integer, ForeignKey('factor_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)
    XR = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'factor'}
    _spec = relationship("FactorSpec", foreign_keys=[uid])


@auto_repr
class EconomicSpec(TimeSeriesSpec):
      __tablename__ = 'economic_spec'

      uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)

      history = Column(Integer,  index=True, nullable=True)
      seasonal_adjustment = Column(Boolean, nullable=False)
      sector = Column(String(100))
      indicator = Column(String(100))
      real = Column(Boolean, nullable=False)

      __mapper_args__ = {'polymorphic_identity': 'economic_spec'}

@auto_repr
class Economic(TimeSeries):
    __tablename__ = 'economic'

    uid = Column(Integer, ForeignKey('economic_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)
    X = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'economic'}
    _spec = relationship("EconomicSpec", foreign_keys=[uid])

    @property
    def _X(self):
        return 'PI'

@auto_repr
class DatabasePickle(Base):
    __tablename__ = 'pickles'

    id = Column(String(255), primary_key=True)
    pickle = Column(LargeBinary(length=2**32-1))




if __name__ == "__main__":

    from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
    session = SessionMgr().getSessionFactory()

