from enum import Enum

class FactorType(Enum):
    Return = 'return'

class FACTOR(Enum):

    # ISG Type Factors
    EQUITY_GLOBAL_ISG = 'cEquity'
    TERM_GLOBAL_ISG = 'cTerm'
    FUNDING_US_ISG = 'cFunding'
    LEVERAGE_US_HKM = 'CLeverage'
    LIQUIDITY_US_PASTOR_STAMBAUGH = 'cLiquidity'
    CARRY_GLOBAL_ISG = 'cFX'
    EQUITY_EMERGING_ISG = 'cEmerging'
    COMMODITY_GLOBAL_ISG = 'cCommodities'
    VALUE_US_ISG = 'HML_US_FF'
    SIZE_US_ISG = 'SMB_US_FF'
    MOMENTUM_US_ISG = 'MOM_US_FF'

    # AQR Factors
    EQUITY_LOW_BETA_US_AQR = 'BAB_US_AQR'
    EQUITY_LOW_BETA_GLOBAL_AQR = 'BAB_GLB_AQR'

    EQUITY_QUALITY_US_AQR = 'QMJ_US_AQR'
    EQUITY_QUALITY_GLOBAL_AQR = 'QMJ_GLB_AQR'

    EQUITY_MARKET_US_AQR = 'MKT_US_AQR'
    EQUITY_MARKET_GLOBAL_AQR = 'MKT_GLB_AQR'

    EQUITY_SIZE_US_AQR = 'SMB_US_AQR'
    EQUITY_SIZE_GLOBAL_AQR = 'SMB_GLB_AQR'

    EQUITY_VALUE_US_AQR = 'HML_US_AQR'
    EQUITY_VALUE_GLOBAL_AQR = 'HML_GLB_AQR'

    EQUITY_MOMENTUM_US_AQR = 'MOM_US_AQR'
    EQUITY_MOMENTUM_GLOBAL_AQR = 'MOM_GLB_AQR'

    # Fama-French Factors
    EQUITY_LOW_BETA_US_FF = 'BAB_US_FF'
    EQUITY_LOW_BETA_GLOBAL_FF = 'BAB_GLB_FF'

    EQUITY_QUALITY_US_FF = 'QMJ_US_FF'
    EQUITY_QUALITY_GLOBAL_FF = 'QMJ_GLB_FF'

    EQUITY_MARKET_US_FF = 'MKT_US_FF'
    EQUITY_MARKET_GLOBAL_FF = 'MKT_GLB_FF'

    EQUITY_SIZE_US_FF = 'SMB_US_FF'
    EQUITY_SIZE_GLOBAL_FF = 'SMB_GLB_FF'

    EQUITY_VALUE_US_FF = 'HML_US_FF'
    EQUITY_VALUE_GLOBAL_FF = 'HML_GLB_FF'

    EQUITY_MOMENTUM_US_FF = 'MOM_US_FF'
    EQUITY_MOMENTUM_GLOBAL_FF = 'MOM_GLB_FF'

    EQUITY_INVESTMENT_US_FF = 'CMA_US_FF'
    EQUITY_INVESTMENT_GLOBAL_FF = 'CMA_GLB_FF'

    EQUITY_PROFITABILITY_US_FF = 'RMW_US_FF'
    EQUITY_PROFITABILITY_GLOBAL_FF = 'RMW_GLB_FF'

    EQUITY_ST_REVERSAL_US_FF = 'STR_US_FF'
    EQUITY_LT_REVERSAL_US_FF = 'LTR_US_FF'

    @staticmethod
    def get_default_return_factor_list():
        return [FACTOR.EQUITY_GLOBAL_ISG,
                FACTOR.TERM_GLOBAL_ISG,
                FACTOR.FUNDING_US_ISG,
                FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH,
                FACTOR.CARRY_GLOBAL_ISG,
                FACTOR.EQUITY_EMERGING_ISG]

    @staticmethod
    def get_default_risk_factor_list():
        return [FACTOR.EQUITY_GLOBAL_ISG,
                FACTOR.TERM_GLOBAL_ISG,
                FACTOR.FUNDING_US_ISG,
                FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH,
                FACTOR.CARRY_GLOBAL_ISG,
                FACTOR.EQUITY_EMERGING_ISG,
                FACTOR.MOMENTUM_US_ISG,
                FACTOR.SIZE_US_ISG,
                FACTOR.VALUE_US_ISG,
                FACTOR.COMMODITY_GLOBAL_ISG]

    @staticmethod
    def get_extended_return_factor_list():
        return [FACTOR.EQUITY_GLOBAL_ISG,
                FACTOR.TERM_GLOBAL_ISG,
                FACTOR.FUNDING_US_ISG,
                FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH,
                FACTOR.EQUITY_EMERGING_ISG]

    @staticmethod
    def get_extended_risk_factor_list():
        return [FACTOR.EQUITY_GLOBAL_ISG,
                FACTOR.TERM_GLOBAL_ISG,
                FACTOR.FUNDING_US_ISG,
                FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH,
                FACTOR.EQUITY_EMERGING_ISG,
                FACTOR.MOMENTUM_US_ISG,
                FACTOR.SIZE_US_ISG,
                FACTOR.VALUE_US_ISG]

