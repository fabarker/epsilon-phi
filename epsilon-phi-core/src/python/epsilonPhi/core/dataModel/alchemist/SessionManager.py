from sqlalchemy import create_engine
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from sqlalchemy.orm import sessionmaker, scoped_session
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.env.Env import DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DRIVER, DB_DATABASE_NAME
from contextlib import contextmanager

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
        return [x[0] for x in session.execute(text("SHOW TABLES IN DEV")).fetchall()]

    def get_all_columns_in_table(self, table_name):
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

    mgr = SessionMgr()
    session = mgr.getSessionFactory()
    eng = mgr.getEngine()
