import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import declarative_base
from epsilonPhi.core.lib.Constants import InfinityTime
from epsilonPhi.core.env.Env import SCHEMA

Base = declarative_base()

class DBOptions(object):
    __table_args__ = {'schema': SCHEMA}

class TemporalMixIn(object):
    modifiedTime = Column(DateTime, nullable=False, default=datetime.datetime.now(), onupdate=datetime.datetime.now())
    fromTime = Column(DateTime, nullable=False, default=datetime.datetime.now())
    toTime = Column(DateTime, nullable=False, default=InfinityTime)


