from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, TypeDecorator, text
from typing import Any
from epsilonPhi.core.lib.Decorators import auto_repr
from sqlalchemy.orm import relationship, declarative_base, declared_attr
from epsilonPhi.core.utils.DateUtils import DateUtils
from decimal import Decimal
import numpy as np
import pandas as pd
import math

Base = declarative_base()

class FloatOrNone(TypeDecorator):
    impl = Float

    def process_bind_param(self, value, dialect) -> Any:
        if isinstance(value, (float, Decimal, np.float64, np.float32)) and math.isnan(float(value)):
                return None
        return value

class DatatypeMapper(object):

    @staticmethod
    def datasource_to_database_mapping(source_datatype):
        if source_datatype.upper() in ['YTM','RY','YTW','IY','RA']:
            return 'RY'
        if source_datatype.upper() in ['DM','DU']:
            return 'DM'
        else:
            return source_datatype

@auto_repr
class TimeSeriesSpec(Base):
    __tablename__ = 'time_series_spec'

    uid = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=True)
    ticker = Column(String(50), nullable=True)
    region = Column(String(100), nullable=True)
    name = Column(String(255), nullable=True)
    category = Column(String(50), nullable=True)
    datasource = Column(String(50), nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'time_series_spec'}


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


############### FX Rates ##############

@auto_repr
class FXRateSpec(TimeSeriesSpec):
    __tablename__ = 'fx_rates_spec'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    foreign_currency = Column(String(3), nullable=False, index=True)
    domestic_currency = Column(String(3), nullable=False, index=True)
    bbid = Column(String(6), nullable=False)
    maturity = Column(String(10), nullable=False)

    __mapper_args__ = {'polymorphic_identity': 'fx_rate_spec'}

    @classmethod
    def get_fx_specs(cls, index_col='uid'):
        from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
        session = SessionMgr().getSessionFactory()
        query = session.query(FXRateSpec.uid, FXRateSpec.bbid, FXRateSpec.maturity, FXRateSpec.provider)
        query_string = query.statement.compile(compile_kwargs={"literal_binds": True}).string
        return pd.read_sql(text(query_string), con=session.get_bind(), index_col=index_col)


@auto_repr
class FXRate(TimeSeries):
    __tablename__ = 'fx_rates'

    uid = Column(Integer, ForeignKey('fx_rates_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    bid = Column(FloatOrNone, nullable=True)
    mid = Column(FloatOrNone, nullable=True)
    ask = Column(FloatOrNone, nullable=True)
    last = Column(FloatOrNone, nullable=True)

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
    def maturity(self):
        return self._spec.maturity

    @classmethod
    def getDataframe(cls, bbids=None, uids=None, index_col='date'):

        from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
        session = SessionMgr().getSessionFactory()

        if bbids is None:
            query = session.query(cls).join(FXRateSpec, cls.uid == FXRateSpec.uid).filter(FXRateSpec.uid.in_(uids))
        else:
            query = session.query(cls).join(FXRateSpec, cls.uid == FXRateSpec.uid).filter(FXRateSpec.bbid.in_(bbids), FXRateSpec.provider != 'BBG')
        query_string = query.statement.compile(compile_kwargs={"literal_binds": True}).string
        return pd.read_sql(text(query_string), con=session.get_bind(), index_col=index_col)

    @classmethod
    def get_spec_df_from_uids(cls, uids, index_col='uid'):
        from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
        session = SessionMgr().getSessionFactory()
        query = session.query(FXRateSpec.uid, FXRateSpec.bbid, FXRateSpec.maturity, FXRateSpec.provider).filter(FXRateSpec.uid.in_(uids))
        query_string = query.statement.compile(compile_kwargs={"literal_binds": True}).string
        return pd.read_sql(text(query_string), con=session.get_bind(), index_col=index_col)

    @classmethod
    def get_spec_df_from_bbids(cls, bbids, index_col='uid'):
        from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
        session = SessionMgr().getSessionFactory()
        query = session.query(FXRateSpec.uid, FXRateSpec.bbid, FXRateSpec.maturity, FXRateSpec.provider).filter(FXRateSpec.bbid.in_(bbids))
        query_string = query.statement.compile(compile_kwargs={"literal_binds": True}).string
        return pd.read_sql(text(query_string), con=session.get_bind(), index_col=index_col)

############### Yield Curves ##############

@auto_repr
class YieldCurveSpec(TimeSeriesSpec):
       __tablename__ = 'yield_curve_spec'

       uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
       currency = Column(String(3), nullable=False, index=True)
       maturity = Column(String(10), nullable=False)
       type = Column(String(10), nullable=False)

       __mapper_args__ = {'polymorphic_identity': 'yield_curve_spec'}

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


@auto_repr
class InterestRate(TimeSeries):
    __tablename__ = 'interest_rate'

    uid = Column(Integer, ForeignKey('interest_rate_spec.uid'), index=True, primary_key=True)
    date = Column(DateTime, primary_key=True)

    IB = Column(FloatOrNone, nullable=True)
    RI = Column(FloatOrNone, nullable=True)
    IR = Column(FloatOrNone, nullable=True)
    IO = Column(FloatOrNone, nullable=True)

    __mapper_args__ = {'polymorphic_identity': 'interest_rate'}
    _spec = relationship("InterestRateSpec", foreign_keys=[uid])

    @property
    def currency(self):
        return self._spec.currency

    @property
    def maturity(self):
        return self._spec.maturity

    @property
    def type(self):
        return self._spec.type


# ############### Implied Volatility ################
#
#
@auto_repr
class ImpliedVolatility(TimeSeries):
    __tablename__ = 'implied_volatility'

    uid = Column(Integer, ForeignKey('time_series_spec.uid'), primary_key=True, index=True)
    date = Column(DateTime, primary_key=True)

    pricing_location = Column(String(3), nullable=True)
    pricing_time = Column(DateTime, nullable=True)

    strike_reference = Column(String(10), nullable=True)
    tenor = Column(String(9), nullable=False, index=True)

    bid = Column(FloatOrNone, nullable=True)
    mid = Column(FloatOrNone, nullable=True)
    ask = Column(FloatOrNone, nullable=True)

    domestic_currency = Column(String(3), nullable=False, index=True)
    foreign_currency = Column(String(3), nullable=False, index=True)
    currency = Column(String(6), nullable=False, index=True)

    __mapper_args__ = {'polymorphic_identity': 'implied_volatility'}

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



if __name__ == "__main__":

    from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
    session = SessionMgr().getSessionFactory()

    info = session.query(TimeSeriesSpec).all()
    for row in info:
        if row.provider == 'BBG':
           row.datasource = 'Bloomberg'
        else:
           row.datasource = 'Datastream'
        session.commit()

        try:
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()




