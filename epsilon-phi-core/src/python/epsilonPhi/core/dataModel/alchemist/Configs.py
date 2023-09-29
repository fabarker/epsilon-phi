from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, TypeDecorator, text
from sqlalchemy.orm import relationship, declarative_base, declared_attr
from epsilonPhi.core.lib.Decorators import auto_repr
from epsilonPhi.core.utils.DateUtils import DateUtils
from decimal import Decimal
from typing import Any
import pandas as pd
import numpy as np
import math

Base = declarative_base()

@auto_repr
class CurrencyConfig(Base):
    __tablename__ = 'currency_config'

    uid = Column(Integer, nullable=True, primary_key=True)
    currency = Column(String(3), nullable=True)

    risk_free_ticker = Column(String(50), nullable=True)
    risk_free_rate = Column(Float, nullable=False)

    inflation_ticker = Column(String(50), nullable=True)
    inflation_rate = Column(Float, nullable=False)

    frequency = Column(String(50), nullable=True)
    dataversion = Column(Float, nullable=False)

    update_time = Column(DateTime, nullable=False)

    __mapper_args__ = {'polymorphic_identity': 'currency_config'}

@auto_repr
class PrivatAssetFlowConfig(Base):
    __tablename__ = 'private_asset_flow_config'

    asOfDate = Column(DateTime, primary_key=True)
    strategy = Column(String(50), primary_key=True)
    year = Column(Integer, nullable=False, primary_key=True)
    type = Column(String(1), nullable=False, primary_key=True)
    value = Column(Float, nullable=False)
    info = Column(String(50), nullable=False)

    __mapper_args__ = {'polymorphic_identity': 'private_asset_flow_config'}


@auto_repr
class SimulationConfig(Base):
    __tablename__ = 'simulation_config'

    asOfDate = Column(DateTime, primary_key=True)
    currency = Column(String(64))

    shortHorizon = Column(Float, server_default='NULL')
    midHorizon = Column(Float, server_default='NULL')
    longHorizon = Column(Float, server_default='NULL')

    number_of_bstraps = Column(Float, server_default='NULL')
    q = Column(Float, server_default='NULL')

    __mapper_args__ = {'polymorphic_identity': 'simulation_config'}

#'@auto_repr
#class FactorConfig(Base):
#    __tablename__ = 'factor_config'
#    asOfDate = Column(DateTime, primary_key=True)

#@auto_repr
#class RiskModel(Base):
#    __tablename__ = 'risk_model'

#    uid = Column(DateTime, primary_key=True)
#    factor = Column(String(100), primary_key=True)
#    frequency = Column(String(100), primary_key=True)

#    __mapper_args__ = {'polymorphic_identity': 'risk_model'}

#@auto_repr
#class ReturnModel(Base):
#    __tablename__ = 'return_model'

#    uid = Column(DateTime, primary_key=True)
#    factor = Column(String(100), primary_key=True)
#    frequency = Column(String(100), primary_key=True)

#   __mapper_args__ = {'polymorphic_identity': 'return_model'}