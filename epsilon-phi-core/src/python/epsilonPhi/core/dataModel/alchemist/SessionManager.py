from epsilonPhi.core.env.Env import DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DRIVER, DB_DATABASE_NAME
from sqlalchemy import create_engine
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from sqlalchemy.orm import sessionmaker, scoped_session
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.Configs import *
from contextlib import contextmanager
import pandas as pd

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

    def is_column_in_table(self, table_name, column_name):
        return column_name in self.get_all_columns_in_table(table_name)

    def is_table_in_database(self, table_name):
        return table_name in self.get_all_tables_in_database()

    def drop_column_from_table(self, table_name, column_name):
        if self.is_table_in_database(table_name) and self.is_column_in_table:
            conn = eng.connect()
            stmt = text("ALTER TABLE " + table_name + " DROP COLUMN " + column_name + ";")
            conn.execute(stmt)
            conn.close()

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

    def get_table_name_from_ticker(self, ticker):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        return self.getSessionFactory().query(TimeSeriesSpec).filter_by(ticker=ticker).first().table_name

    def get_table_name_from_uid(self, uid):
        from epsilonPhi.core.dataModel.alchemist.DataModel import TimeSeriesSpec
        ts_info = self.getSessionFactory().query(TimeSeriesSpec).filter_by(uid=uid).first()
        return ts_info._TimeSeriesSpec__map.table_name

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
        query = self.getSessionFactory().query(class_).filter(class_.uid.in_([uid]))
        return self.query_format_df(query)

    def get_time_series_spec_from_uid(self, uid: str, return_df=False):

        q = self.getSessionFactory().query(TimeSeriesSpec).filter(TimeSeriesSpec.uid == uid)

        if return_df:
           return self.query_format_df(q)
        else:
           return q.first()

    def get_time_series_spec_from_ticker(self, ticker: str, return_df=False):

        q = self.getSessionFactory().query(TimeSeriesSpec).filter(TimeSeriesSpec.ticker == ticker)

        if return_df:
           return self.query_format_df(q)
        else:
           return q.first()

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

    session = SessionMgr().getSessionFactory()

    SQL_1 = "SELECT * FROM `epsilon-phi-dev`.time_series_spec where name like '%Spot%' and provider = 'GS' and category = 'Implied Volatility'"
    SQL_2 = "SELECT * FROM `epsilon-phi-dev`.time_series_spec where name like '%ATMF%' and provider = 'GS' and category = 'Implied Volatility'"

    df_ATM = pd.read_sql(SQL_2, session.get_bind())
    df_Spt = pd.read_sql(SQL_2, session.get_bind())
    df = pd.concat((df_ATM, df_Spt), axis=0)

    table_name = '`epsilon-phi-dev`.implied_volatility'
    unique_uids = df_ATM.uid.unique()
    N = len(unique_uids)

    ctr = 0
    for uid in df_ATM.uid.unique():

        print(N-ctr)
        READ_STATEMENT = f"SELECT * FROM {table_name} WHERE uid = {uid};"
        df_data = pd.read_sql(READ_STATEMENT, session.get_bind())
        df_data.relative_strike = '100'

        DELETE_STATEMENT = f"DELETE FROM {table_name} WHERE uid = {uid};"
        with session.get_bind().begin() as conn:
            conn.execute(text(DELETE_STATEMENT))

        df_data.to_sql('implied_volatility', session.get_bind(), if_exists='append', index=False)
        ctr = ctr + 1


    try:
        session.query(ImpliedVolatility).filter(ImpliedVolatility.relative_strike.in_(['Spot', 'ATMF'])).update(
            {"relative_strike": "100"})
        session.commit()
        print("UPDATE successful.")
    except Exception as e:
        session.rollback()
        print("Error occurred during UPDATE:", str(e))
    finally:
        session.close()
