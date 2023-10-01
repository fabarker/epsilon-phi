from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Float
from epsilonPhi.core.lib.Decorators import auto_repr
from epsilonPhi.core.dataModel.alchemist.BaseData import Base, TemporalMixIn
from sqlalchemy.types import TypeDecorator, String
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency

class FrequencyType(TypeDecorator):
    impl = String
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return value.value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return Frequency(value)


@auto_repr
class CurrencyConfig(Base, TemporalMixIn):
    __tablename__ = 'currency_config'

    uid = Column(Integer, nullable=True, primary_key=True)
    currency = Column(String(3), nullable=True, primary_key=True)

    risk_free_ticker = Column(String(50), nullable=True)
    risk_free_rate = Column(Float, nullable=False)

    inflation_ticker = Column(String(50), nullable=True)
    inflation_rate = Column(Float, nullable=False)

    frequency = Column(FrequencyType(50), nullable=True, primary_key=True)
    dataversion = Column(Integer, nullable=False, primary_key=True)

    __mapper_args__ = {'polymorphic_identity': 'currency_config'}

@auto_repr
class PrivatAssetFlowConfig(Base, TemporalMixIn):
    __tablename__ = 'private_asset_flow_config'

    asOfDate = Column(DateTime, primary_key=True)
    strategy = Column(String(50), primary_key=True)
    year = Column(Integer, nullable=False, primary_key=True)
    type = Column(String(1), nullable=False, primary_key=True)
    value = Column(Float, nullable=False)
    info = Column(String(50), nullable=False)

    __mapper_args__ = {'polymorphic_identity': 'private_asset_flow_config'}
@auto_repr
class EstimationConfig(Base, TemporalMixIn):
    __tablename__ = 'estimation_config'

    uid = Column(Integer, nullable=True, primary_key=True)
    currency = Column(String(64))

    __mapper_args__ = {'polymorphic_identity': 'estimation_config'}

@auto_repr
class CrisisConfig(Base, TemporalMixIn):
    __tablename__ = 'crises_config'

    crisis_id = Column(BigInteger, index=True, primary_key=True)
    crisis_name = Column(String(128))
    crisis_start_date = Column(DateTime)
    crisis_end_date = Column(DateTime)
    __mapper_args__ = {'polymorphic_identity': 'crises'}

@auto_repr
class SimulationConfig(Base, TemporalMixIn):
    __tablename__ = 'simulation_config'

    uid = Column(Integer, nullable=True, primary_key=True)

    currency = Column(String(64))
    dataversion = Column(Float, nullable=False, primary_key=True)

    shortHorizon = Column(Float, nullable=True)
    midHorizon = Column(Float, nullable=True)
    longHorizon = Column(Float, nullable=True)
    number_of_bstraps = Column(Float, nullable=True)
    q = Column(Float, nullable=True)

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

#@auto_repr
#class FactorConfig(Base, TemporalMixIn):
#    __tablename__ = 'factor_config'
#    asOfDate = Column(DateTime, primary_key=True)

#@auto_repr
#class RiskModelConfig(Base, TemporalMixIn):
#    __tablename__ = 'risk_model_config'

#    uid = Column(DateTime, primary_key=True)
#    factor = Column(String(100), primary_key=True)
#    frequency = Column(String(100), primary_key=True)

#    __mapper_args__ = {'polymorphic_identity': 'risk_model'}

#@auto_repr
#class ReturnModelConfig(Base, TemporalMixIn):
#    __tablename__ = 'return_model_config'

#    uid = Column(DateTime, primary_key=True)
#    factor = Column(String(100), primary_key=True)
#    frequency = Column(String(100), primary_key=True)

#    __mapper_args__ = {'polymorphic_identity': 'return_model'}


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
    mgr = SessionMgr.instance
