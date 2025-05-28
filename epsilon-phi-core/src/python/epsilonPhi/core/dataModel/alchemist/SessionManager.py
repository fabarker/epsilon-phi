from epsilonPhi.core.env.Env import DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DRIVER, DB_DATABASE_NAME
from sqlalchemy import create_engine, exists, distinct
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from sqlalchemy.orm import sessionmaker, scoped_session
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.Configs import *
from contextlib import contextmanager
from sqlalchemy import distinct, func
import pickle
import pandas as pd

import os
os.environ['APPDATA'] = ""

tblToEngine = {
    'AssetConfig': 'PWM_DEFAULT_ENGINE',
    'AssetCurrencyConfig': 'PWM_DEFAULT_ENGINE'}

@SingletonDecorator
class SessionMgr(object):
    _engineCache = dict()
    _sessionCache = dict()
    _defaultEngine = None

    def __init__(self):
        self.initalize()
    def get_connection_string(self, database_name=DB_DATABASE_NAME):
        return DB_DRIVER + '://' + DB_USERNAME +\
            ':' + DB_PASSWORD + '@' +\
            DB_HOSTNAME + '/' + database_name

    def create_engine(self, database_name):
        self._engineCache[database_name] = create_engine(self.get_connection_string(database_name))
        Base.metadata.create_all(self._engineCache[database_name])

    def initalize(self):
        self.create_engine(DB_DATABASE_NAME)

    def getSession(self, database_name=None):

        if database_name is None:
            database_name = DB_DATABASE_NAME

        if database_name in self._sessionCache.keys():
           session, _ = self._sessionCache.get(database_name)
           return session
        else:
            session_factory = sessionmaker(bind=self.getEngine())
            Session = scoped_session(session_factory)
            self._sessionCache[database_name] = (session_factory, Session)
            return self.getSession(database_name)

    def getSessionFactory(self, database_name=None):

        if database_name is None:
            database_name = DB_DATABASE_NAME

        if database_name in self._sessionCache.keys():
           _, sessionFactory = self._sessionCache.get(database_name)
           return sessionFactory
        else:
            session_factory = sessionmaker(bind=self.getEngine())
            Session = scoped_session(session_factory)
            self._sessionCache[database_name] = (session_factory, Session)
            return self.getSessionFactory(database_name)
    def getEngine(self, database_name=None):

        if database_name is None:
            database_name = DB_DATABASE_NAME

        if database_name in self._engineCache.keys():
            return self._engineCache.get(database_name)
        else:
            self.create_engine(database_name)
            return self.getEngine(database_name)

    def get_all_tables_in_database(self):
        session = self.getSessionFactory()
        return [x[0] for x in session.execute(text("SHOW TABLES")).fetchall()]

    def get_all_columns_in_table(self, table_name):
        session = self.getSessionFactory()
        if self.is_table_in_database(table_name):
            return [x[0] for x in session.execute(text("SHOW COLUMNS IN " + table_name)).fetchall()]
        else:
            return []

    def get_latest_observation_date_from_tickers(self, ticker):
        session = self.getSessionFactory()

        table_name = self.get_table_name_from_ticker(ticker)
        table = self.fetch_model_class_from_table_name(table_name)

        res = session.query(TimeSeriesSpec.symbol,
                            func.max(table.date)).join(table,
                                                       TimeSeriesSpec.uid == table.uid).\
                                                       filter(TimeSeriesSpec.symbol.in_([ticker])).\
                                                       group_by(TimeSeriesSpec.symbol).all()
        return res[0][1]

    def get_bbid_from_region(self, region):
        return self.getSessionFactory().query(CurrencyMapper.code).filter(CurrencyMapper.region == region).scalar()

    def get_region_from_currency(self, currency):
        return self.getSessionFactory().query(CurrencyMapper.region).filter(CurrencyMapper.code == currency.upper()).scalar()

    def load_currency_mapping(self):
        return self.getSessionFactory().query(CurrencyMapper).all()

    def is_EURO_legacy(self, currency):
        return self.getSessionFactory().query(CurrencyMapper.EUR_legacy).filter(
            CurrencyMapper.code == currency.upper()).scalar()

    def get_currency_from_region(self, region):
        return self.getSessionFactory().query(CurrencyMapper.code).filter(CurrencyMapper.region == region).scalar()

    def get_currency_name(self, code):
        return self.getSessionFactory().query(CurrencyMapper.name).filter(CurrencyMapper.code == code).scalar()

    def is_column_in_table(self, table_name, column_name):
        return column_name in self.get_all_columns_in_table(table_name)

    def is_table_in_database(self, table_name):
        return table_name in self.get_all_tables_in_database()

    def fetch_model_class_from_table_name(self, table_name):
        for mapper in Base.registry.mappers:
            if hasattr(mapper, 'class_') and mapper.class_.__tablename__ == table_name.lower():
                return mapper.class_

    def get_ticker_table_mapping(self, tickers: list):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec, CategoryTableMapping
        return pd.DataFrame(self.getSessionFactory().query(TimeSeriesSpec.ticker, CategoryTableMapping.table_name)\
            .join(CategoryTableMapping, TimeSeriesSpec.category == CategoryTableMapping.category)\
            .filter(TimeSeriesSpec.ticker.in_(tickers))\
            .all()).set_index('table_name', drop=True)

    def get_ticker_spec_table_mapping(self, tickers: list):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec, CategoryTableMapping
        return pd.DataFrame(self.getSessionFactory().query(TimeSeriesSpec.ticker, CategoryTableMapping.spec_table_name) \
                            .join(CategoryTableMapping, TimeSeriesSpec.category == CategoryTableMapping.category) \
                            .filter(TimeSeriesSpec.ticker.in_(tickers)) \
                            .all()).set_index('spec_table_name', drop=True)

    def get_table_name_from_ticker(self, ticker):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        return self.getSessionFactory().query(TimeSeriesSpec).filter_by(ticker=ticker).first()._TimeSeriesSpec__map.table_name

    def get_spec_table_name_from_ticker(self, ticker):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        return self.getSessionFactory().query(TimeSeriesSpec).filter_by(ticker=ticker).first()._TimeSeriesSpec__map.spec_table_name

    def get_table_name_from_uid(self, uid):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        ts_info = self.getSessionFactory().query(TimeSeriesSpec).filter_by(uid=uid).first()
        return ts_info._TimeSeriesSpec__map.table_name

    def get_spec_table_name_from_uid(self, uid):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        ts_info = self.getSessionFactory().query(TimeSeriesSpec).filter_by(uid=uid).first()
        return ts_info._TimeSeriesSpec__map.spec_table_name

    def get_ticker_from_uid(self, uid: int) -> str:
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        return self.getSessionFactory().query(TimeSeriesSpec).filter_by(uid=uid).first().ticker

    def get_uid_from_ticker(self, ticker: str) -> int:
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        return self.getSessionFactory().query(TimeSeriesSpec).filter_by(ticker=ticker).first().uid

    def fetch_model_class_from_uid(self, uid: int):
        table_name = self.get_table_name_from_uid(uid)
        return self.fetch_model_class_from_table_name(table_name)

    def fetch_model_class_from_ticker(self, ticker: str):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        table_name = self.getSessionFactory().query(TimeSeriesSpec).filter_by(ticker=ticker).first().table_name
        return self.fetch_model_class_from_table_name(table_name)

    @staticmethod
    def query_format_df(query):
        q = query.statement.compile(compile_kwargs={"literal_binds": True}).string
        return pd.read_sql(q.replace('"', ''), query.session.get_bind())

    def get_dataframe_from_uid(self, uid: int):
        class_ = self.fetch_model_class_from_uid(uid)
        cols = [x.label(class_()._X) if x.name == 'X' else x for x in class_.__table__._columns.values()]
        query = self.getSessionFactory().query(*cols).filter(class_.uid.in_([uid]))

        return self.query_format_df(query)

    def get_time_series_spec_from_uid(self, uid: str, return_df=False):

        table_name = self.get_spec_table_name_from_uid(uid)
        class_ = self.fetch_model_class_from_table_name(table_name)
        q = self.getSessionFactory().query(class_).filter(class_.uid == uid)

        if return_df:
           return self.query_format_df(q)
        else:
           return q.first()

    def get_time_series_spec_from_ticker(self, ticker: str, return_df=False):

        table_name = self.get_spec_table_name_from_ticker(ticker)
        class_ = self.fetch_model_class_from_table_name(table_name)
        q = self.getSessionFactory().query(class_).filter(class_.ticker == ticker)

        if return_df:
           return self.query_format_df(q)
        else:
           return q.first()


    def get_time_series_currency(self, ticker_uid):

        if isinstance(ticker_uid, str):
           spec = self.get_time_series_spec_from_ticker(ticker_uid)
        else:
           spec = self.get_time_series_spec_from_uid(ticker_uid)
        return spec.denominated_currency, spec.exposure_currency, spec.hedge_ratio

    def get_imf_code_from_region(self, region):
        return self.getSessionFactory().query(CurrencyMapper.code_imf).filter(
            CurrencyMapper.region == region).scalar()

    def get_imf_code_from_currency(self, currency):
        return self.getSessionFactory().query(CurrencyMapper.code_imf).filter(
            CurrencyMapper.currency == currency).scalar()

    def get_factor_ticker(self, factor_mnemonic, region=None, universe=None, provider=None):
        q = self.getSessionFactory().query(FactorSpec.ticker).filter(FactorSpec.factor == factor_mnemonic)

        if region is not None:
            q = q.filter(FactorSpec.region == region)
        if universe is not None:
            q = q.filter(FactorSpec.universe == universe)
        if provider is not None:
            q = q.filter(FactorSpec.provider == provider)
        res = q.all()

        if len(res) == 1:
            return res[0][0]
        elif len(res) > 1:
            return [x[0] for x in q.all()]
        else:
            return None

    def get_max_uid(self):
        from sqlalchemy import func
        return self.getSessionFactory().query(func.max(TimeSeriesSpec.uid)).scalar()

    def get_inflation_rates_for_region(self, region):
        pass

    def get_yield_curve_tickers_for_region(self, region):

        q = self.getSessionFactory().query(YieldCurveSpec.uid,
                                           YieldCurveSpec.ticker,
                                           YieldCurveSpec.maturity,
                                           YieldCurveSpec.type).filter(YieldCurveSpec.region.in_([region]))
        return self.query_format_df(q).set_index('uid')

    def get_govt_bond_tickers_for_region(self, region, maturity):
        session = self.getSessionFactory()
        return session.query(BondIndexSpec.ticker).filter(BondIndexSpec.region == region,
                                                BondIndexSpec.sector == 'Govt',
                                                BondIndexSpec.maturity == maturity).scalar()

    def get_interest_rates_for_region(self, region, maturity=None, type=None):

        spec = self.get_interest_rate_tickers_from_region(region, maturity=maturity, type=type)
        q = self.getSessionFactory().query(InterestRate).filter(InterestRate.uid.in_(spec.get('uid').values.flatten()))
        res = self.query_format_df(q).drop(columns=['IB', 'RI', 'IO'])

        if res.size > 0:
            res.set_index(['uid', 'date'], inplace=True)
            df = res.mean(axis=1).to_frame('IR')

            panel = pd.concat([df.loc[x] for x in np.unique(df.index.get_level_values(0))], axis=1)
            panel.columns = pd.MultiIndex.from_frame(spec)
            return panel.dropna(axis=1, how='all')
        else:
            return pd.DataFrame()

    def get_consumer_price_index_tickers_from_region(self, region):

        q = self.getSessionFactory().query(EconomicSpec.uid,
                                           EconomicSpec.ticker).filter(EconomicSpec.region.in_([region]),
                                                                       EconomicSpec.indicator == 'CPI',
                                                                       EconomicSpec.seasonal_adjustment == 1)

        return self.query_format_df(q)

    def get_interest_rate_maturities_for_region(self, region):
        q = self.getSessionFactory().query(distinct(InterestRateSpec.maturity)).filter(InterestRateSpec.region.in_([region])).all()
        if len(q) > 0:
            return [x[0] for x in q]

    def get_interest_rate_tickers_from_region(self, region, maturity=None, type=None):

        q = self.getSessionFactory().query(InterestRateSpec.uid,
                                               InterestRateSpec.ticker,
                                               InterestRateSpec.maturity,
                                               InterestRateSpec.type).filter(InterestRateSpec.region.in_([region]))

        if maturity is not None:
           if not DateUtils.is_iterable(maturity):
                maturity = [maturity]
           q = q.filter(InterestRateSpec.maturity.in_(maturity))

        if type is not None:
           if not DateUtils.is_iterable(type):
                type = [type]
           q = q.filter(InterestRateSpec.type.in_(type))

        return self.query_format_df(q)

    def save_pickle_to_database(self, pickle_object, pickle_id):
        print('Saving {} to pickles'.format(pickle_id))

        new_row = DatabasePickle(pickle=pickle_object, id=pickle_id)

        session = self.getSessionFactory()
        session.add(new_row)
        session.commit()
        session.close()

    def delete_pickle_from_database(self, pickle_id):

        print('Delete {} from pickles'.format(pickle_id))
        session = self.getSessionFactory()
        session.query(DatabasePickle).filter(DatabasePickle.id == pickle_id).delete()
        session.commit()
        session.close()


    def pickle_and_save_to_database(self, obj, id):

        print('Saving {} to pickles'.format(id))

        pickled_df = pickle.dumps(obj)
        new_row = DatabasePickle(pickle=pickled_df, id=id)

        session = self.getSessionFactory()
        session.add(new_row)
        session.commit()
        session.close()

    def load_pickle_from_database(self, uid):

        print('Loading {} from pickles'.format(uid))

        session = self.getSessionFactory()
        pickled_obj = session.query(DatabasePickle).filter_by(id=uid).first().pickle
        session.close()
        return pickle.loads(pickled_obj)

    def is_pickled(self, id):
        return self.getSessionFactory().query(exists().where(DatabasePickle.id == id)).scalar()

    def show_table(self, table):
        import pandasgui
        pandasgui.show(self.query_format_df(self.getSessionFactory().query(table)))

    def get_underlier_asset_class(self, underlier):
        return (self.getSessionFactory().query(ImpliedVolatilitySpec.category).
                filter(ImpliedVolatilitySpec.security == underlier).scalar())

    def load_FX_vol_surface_from_underlier(self, underlier, pricing_location=None):


        print('Loading Vol Surface Data for Security {}'.format(underlier))
        _path = '/Users/francisbarker/Desktop/ivol cache/' + underlier + '.csv'
        if os.path.exists(_path):
           df_ = pd.read_csv(_path)
           df_.date = pd.to_datetime(df_.date)
        else:
           session = self.getSessionFactory()
           q = session.query(ImpliedVolatilityNew).filter(ImpliedVolatilityNew.security ==
                                                           underlier)

           if pricing_location is None:
              pricing_location = 'NYC'
           q = q.filter(ImpliedVolatilityNew.pricing_location == pricing_location)

           # load dataframe
           frame = self.query_format_df(q).dropna(how='all', axis=1)
           df = frame[frame.strike_reference == 'delta']
           df = df.replace('DN', '-99900')
           df['relative_strike'] = df['relative_strike'].astype(float).astype(int) / 100

           # Drop duplicates
           dduped = df.drop_duplicates(subset=['date', 'tenor', 'relative_strike'])

           # Make sure no weekends by accident
           wkends = pd.to_datetime(dduped.date.values).dayofweek.isin([5, 6])
           df_ = dduped[~wkends].copy()

           # Set the date formats in the frame
           df_['date'] = pd.to_datetime(df_.get('date').values)
           df_['t'] = DateUtils.Rdate_to_mat(df_['tenor'].values)
           df_.to_csv(_path, index=False)
        return df_


    def get_ivols(self, underlier, pricing_location=None, index=None):

        asset_class = self.get_underlier_asset_class(underlier)

        if asset_class == 'FX':
           df = self.load_FX_vol_surface_from_underlier(underlier, pricing_location)
        else:
            raise ValueError('Error - underlier {} not recognised'.format(underlier))

        if index is not None:
            return df.set_index(index, drop=True)
        else:
            return df.copy()

    def get_ivol_spec(self, underlier):
        session = self.getSessionFactory()
        _q = (session.query(ImpliedVolatilitySpec).
              filter(ImpliedVolatilitySpec.security == underlier))
        return self.query_format_df(_q).T.to_dict().get(0)


@contextmanager
def session_scope():
    scoped_session = SessionMgr().getSessionFactory()
    session = scoped_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":

    sessionMgr = SessionMgr()
    session = sessionMgr.getSessionFactory()

    pickle_ids = session.query(DatabasePickle.id).all()

    for id in pickle_ids:
        if 'GS' in id[0]:
            sessionMgr.delete_pickle_from_database(id[0])


