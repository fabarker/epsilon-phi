"""Financial time-series schema (metadata JTI + per-type observation tables).

Timestamp contract: every ``date`` / ``DateTime`` column stores a naive
datetime in UTC. Callers MUST normalize to UTC and strip tzinfo before
persisting; values are returned naive and are to be interpreted as UTC.
``EconomicData.release_date`` is a UTC calendar date (publication vintage).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    TypeDecorator,
    inspect,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    selectin_polymorphic,
    with_polymorphic,
)


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class NullableNumeric(TypeDecorator[Decimal]):
    """DECIMAL(28, 10) that stores NaN/Infinity as NULL and returns Decimal."""

    impl = Numeric(28, 10)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Decimal | None:
        if value is None:
            return None

        try:
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

        return decimal_value if decimal_value.is_finite() else None

    def process_result_value(self, value: Any, dialect: Any) -> Decimal | None:
        if value is None:
            return None

        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        return decimal_value if decimal_value.is_finite() else None


NUM = NullableNumeric


class CurrencyMixin:
    denominated_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    exposure_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    hedge_ratio: Mapped[Decimal | None] = mapped_column(NUM)


def _spec_uid() -> Mapped[int]:
    return mapped_column(
        Integer,
        ForeignKey("time_series_spec.uid", ondelete="CASCADE"),
        primary_key=True,
    )


def _data_uid(spec_table_name: str) -> Mapped[int]:
    return mapped_column(
        Integer,
        ForeignKey(f"{spec_table_name}.uid", ondelete="CASCADE"),
        primary_key=True,
    )


# =============================================================================
# Metadata: joined-table inheritance
# =============================================================================


class TimeSeriesSpec(Base):
    __tablename__ = "time_series_spec"

    uid: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)

    ticker: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    datasource: Mapped[str] = mapped_column(String(50), nullable=False)

    # Kept as plain business metadata. It is no longer used for ORM resolution.
    category: Mapped[str | None] = mapped_column(String(50))
    provider: Mapped[str | None] = mapped_column(String(250))
    region: Mapped[str | None] = mapped_column(String(100))
    symbol: Mapped[str | None] = mapped_column(String(50))
    frequency: Mapped[str | None] = mapped_column(String(10))

    __table_args__ = (
        Index("ix_time_series_spec_ticker", "ticker"),
        Index("ix_time_series_spec_asset_type", "asset_type"),
        Index("ix_time_series_spec_lookup", "datasource", "ticker", "frequency"),
    )

    __mapper_args__ = {
        "polymorphic_on": asset_type,
        "polymorphic_identity": "base",
    }


class EquitySpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "equity_spec"

    uid: Mapped[int] = _spec_uid()

    isin: Mapped[str] = mapped_column("ISIN", String(100), nullable=False)
    sedol: Mapped[str] = mapped_column("SEDOL", String(100), nullable=False)

    exchange: Mapped[str] = mapped_column(String(100), nullable=False)
    exchange_code: Mapped[str | None] = mapped_column(String(20))
    exchange_mnemonic: Mapped[str | None] = mapped_column(String(20))

    sector: Mapped[str | None] = mapped_column(String(255))
    industry_group: Mapped[str | None] = mapped_column(String(255))
    industry: Mapped[str | None] = mapped_column(String(255))
    sub_industry: Mapped[str | None] = mapped_column(String(255))

    data: Mapped[list["EquityData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_equity_spec_isin", "ISIN"),
        Index("ix_equity_spec_sedol", "SEDOL"),
        Index("ix_equity_spec_exchange", "exchange"),
        Index("ix_equity_spec_sector", "sector"),
    )

    __mapper_args__ = {
        "polymorphic_identity": "equity",
        "polymorphic_load": "selectin",
    }


class ETFSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "etf_spec"

    uid: Mapped[int] = _spec_uid()

    isin: Mapped[str] = mapped_column("ISIN", String(100), nullable=False)
    sedol: Mapped[str] = mapped_column("SEDOL", String(100), nullable=False)

    exchange: Mapped[str] = mapped_column(String(100), nullable=False)
    exchange_code: Mapped[str | None] = mapped_column(String(20))
    exchange_mnemonic: Mapped[str | None] = mapped_column(String(20))

    data: Mapped[list["ETFData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_etf_spec_isin", "ISIN"),
        Index("ix_etf_spec_sedol", "SEDOL"),
        Index("ix_etf_spec_exchange", "exchange"),
    )

    __mapper_args__ = {
        "polymorphic_identity": "etf",
        "polymorphic_load": "selectin",
    }


class EquityIndexSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "equity_index_spec"

    uid: Mapped[int] = _spec_uid()

    data: Mapped[list["EquityIndexData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __mapper_args__ = {
        "polymorphic_identity": "equity_index",
        "polymorphic_load": "selectin",
    }


class BondIndexSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "bond_index_spec"

    uid: Mapped[int] = _spec_uid()

    sector: Mapped[str] = mapped_column(String(20), nullable=False)
    rating: Mapped[str] = mapped_column(String(10), nullable=False)
    maturity_band: Mapped[str] = mapped_column(String(20), nullable=False)
    maturity: Mapped[int | None] = mapped_column(Integer)

    data: Mapped[list["BondIndexData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_bond_index_spec_rating", "rating"),
        Index("ix_bond_index_spec_sector", "sector"),
        Index("ix_bond_index_spec_maturity_band", "maturity_band"),
    )

    __mapper_args__ = {
        "polymorphic_identity": "bond_index",
        "polymorphic_load": "selectin",
    }


class CommodityIndexSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "commodity_index_spec"

    uid: Mapped[int] = _spec_uid()

    data: Mapped[list["CommodityIndexData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __mapper_args__ = {
        "polymorphic_identity": "commodity_index",
        "polymorphic_load": "selectin",
    }


class FXRateSpec(TimeSeriesSpec):
    __tablename__ = "fx_rates_spec"

    uid: Mapped[int] = _spec_uid()

    foreign_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    domestic_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    bbid: Mapped[str] = mapped_column(String(6), nullable=False)
    maturity: Mapped[str] = mapped_column(String(10), nullable=False)
    fx_type: Mapped[str] = mapped_column("type", String(1), nullable=False)

    data: Mapped[list["FXRateData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_fx_rates_spec_pair", "foreign_currency", "domestic_currency"),
        Index("ix_fx_rates_spec_bbid", "bbid"),
    )

    __mapper_args__ = {
        "polymorphic_identity": "fx_rate",
        "polymorphic_load": "selectin",
    }


class YieldCurveSpec(TimeSeriesSpec):
    __tablename__ = "yield_curve_spec"

    uid: Mapped[int] = _spec_uid()

    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    curve_type: Mapped[str] = mapped_column("type", String(10), nullable=False)

    data: Mapped[list["YieldCurveData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_yield_curve_spec_currency", "currency"),)

    __mapper_args__ = {
        "polymorphic_identity": "yield_curve",
        "polymorphic_load": "selectin",
    }


class InterestRateSpec(TimeSeriesSpec):
    __tablename__ = "interest_rate_spec"

    uid: Mapped[int] = _spec_uid()

    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    maturity: Mapped[str] = mapped_column(String(10), nullable=False)
    rate_type: Mapped[str] = mapped_column("type", String(10), nullable=False)

    data: Mapped[list["InterestRateData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_interest_rate_spec_currency_maturity", "currency", "maturity"),)

    __mapper_args__ = {
        "polymorphic_identity": "interest_rate",
        "polymorphic_load": "selectin",
    }


class HedgeFundIndexSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "hedge_fund_index_spec"

    uid: Mapped[int] = _spec_uid()

    strategy_type: Mapped[str] = mapped_column(String(150), nullable=False)

    data: Mapped[list["HedgeFundIndexData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_hedge_fund_index_spec_strategy_type", "strategy_type"),)

    __mapper_args__ = {
        "polymorphic_identity": "hedge_fund_index",
        "polymorphic_load": "selectin",
    }


class FutureSpec(TimeSeriesSpec, CurrencyMixin):
    __tablename__ = "future_spec"

    uid: Mapped[int] = _spec_uid()

    future: Mapped[str | None] = mapped_column(String(30))
    start_date: Mapped[dt.datetime | None] = mapped_column(DateTime)
    tick_size: Mapped[Decimal | None] = mapped_column(NUM)
    tick_value: Mapped[Decimal | None] = mapped_column(NUM)
    contract_size: Mapped[Decimal | None] = mapped_column(NUM)
    cycle: Mapped[str | None] = mapped_column(String(100))

    exchange: Mapped[str] = mapped_column(String(100), nullable=False)
    exchange_name: Mapped[str | None] = mapped_column(String(100))

    security: Mapped[str] = mapped_column(String(100), nullable=False)
    security_name: Mapped[str | None] = mapped_column(String(100))
    security_type: Mapped[str | None] = mapped_column(String(100))
    security_unit: Mapped[str | None] = mapped_column(String(100))

    position_forward: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    data: Mapped[list["FutureData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_future_spec_exchange", "exchange"),
        Index("ix_future_spec_security", "security"),
    )

    __mapper_args__ = {
        "polymorphic_identity": "future",
        "polymorphic_load": "selectin",
    }


class FactorSpec(TimeSeriesSpec):
    __tablename__ = "factor_spec"

    uid: Mapped[int] = _spec_uid()

    factor: Mapped[str] = mapped_column(String(100), nullable=False)
    universe: Mapped[str | None] = mapped_column(String(100))

    data: Mapped[list["FactorData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_factor_spec_factor_universe", "factor", "universe"),)

    __mapper_args__ = {
        "polymorphic_identity": "factor",
        "polymorphic_load": "selectin",
    }


class EconomicSpec(TimeSeriesSpec):
    __tablename__ = "economic_spec"

    uid: Mapped[int] = _spec_uid()

    history: Mapped[int | None] = mapped_column(Integer)
    seasonal_adjustment: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sector: Mapped[str | None] = mapped_column(String(100))
    indicator: Mapped[str] = mapped_column(String(100), nullable=False)
    real: Mapped[bool] = mapped_column(Boolean, nullable=False)

    data: Mapped[list["EconomicData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_economic_spec_indicator", "indicator"),)

    __mapper_args__ = {
        "polymorphic_identity": "economic",
        "polymorphic_load": "selectin",
    }


class ImpliedVolatilitySpec(TimeSeriesSpec):
    __tablename__ = "implied_volatility_spec"

    uid: Mapped[int] = _spec_uid()

    security: Mapped[str] = mapped_column(String(100), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)

    data: Mapped[list["ImpliedVolatilityData"]] = relationship(
        back_populates="spec", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_implied_volatility_spec_security", "security"),)

    __mapper_args__ = {
        "polymorphic_identity": "implied_volatility",
        "polymorphic_load": "selectin",
    }


# =============================================================================
# Observations: concrete per-type tables, resolved through registry
# =============================================================================


class ObservationBase(Base):
    __abstract__ = True

    date: Mapped[dt.datetime] = mapped_column(DateTime, primary_key=True)


class EquityData(ObservationBase):
    __tablename__ = "equity"

    uid: Mapped[int] = _data_uid("equity_spec")

    price_index: Mapped[Decimal | None] = mapped_column("PI", NUM)
    return_index: Mapped[Decimal | None] = mapped_column("RI", NUM)
    dividend_yield: Mapped[Decimal | None] = mapped_column("DY", NUM)
    market_value: Mapped[Decimal | None] = mapped_column("MV", NUM)

    pe: Mapped[Decimal | None] = mapped_column("PE", NUM)
    eps: Mapped[Decimal | None] = mapped_column("EPS", NUM)
    price_to_book_value: Mapped[Decimal | None] = mapped_column("PTBV", NUM)
    volume: Mapped[Decimal | None] = mapped_column("VO", NUM)
    eps_est_12m: Mapped[Decimal | None] = mapped_column("EPS1FD12", NUM)
    shares_outstanding: Mapped[Decimal | None] = mapped_column("NOSH", NUM)
    sales: Mapped[Decimal | None] = mapped_column("WC01001", NUM)
    adjusted_price_close: Mapped[Decimal | None] = mapped_column("APC", NUM)
    earnings_yield_est_12m: Mapped[Decimal | None] = mapped_column("529E", NUM)
    eps_est_12m_ds: Mapped[Decimal | None] = mapped_column("DIEP", NUM)
    pe_est_12m: Mapped[Decimal | None] = mapped_column("DIPE", NUM)

    ask: Mapped[Decimal | None] = mapped_column("PA", NUM)
    bid: Mapped[Decimal | None] = mapped_column("PB", NUM)
    low: Mapped[Decimal | None] = mapped_column("PL", NUM)
    high: Mapped[Decimal | None] = mapped_column("PH", NUM)

    spec: Mapped[EquitySpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_equity_date_uid", "date", "uid"),)


class ETFData(ObservationBase):
    __tablename__ = "etf"

    uid: Mapped[int] = _data_uid("etf_spec")

    price_index: Mapped[Decimal | None] = mapped_column("PI", NUM)
    return_index: Mapped[Decimal | None] = mapped_column("RI", NUM)
    dividend_yield: Mapped[Decimal | None] = mapped_column("DY", NUM)
    market_value: Mapped[Decimal | None] = mapped_column("MV", NUM)

    volume: Mapped[Decimal | None] = mapped_column("VO", NUM)
    shares_outstanding: Mapped[Decimal | None] = mapped_column("NOSH", NUM)

    ask: Mapped[Decimal | None] = mapped_column("PA", NUM)
    bid: Mapped[Decimal | None] = mapped_column("PB", NUM)
    low: Mapped[Decimal | None] = mapped_column("PL", NUM)
    high: Mapped[Decimal | None] = mapped_column("PH", NUM)

    spec: Mapped[ETFSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_etf_date_uid", "date", "uid"),)


class EquityIndexData(ObservationBase):
    __tablename__ = "equity_index"

    uid: Mapped[int] = _data_uid("equity_index_spec")

    dividend_yield: Mapped[Decimal | None] = mapped_column("DY", NUM)
    return_index: Mapped[Decimal | None] = mapped_column("RI", NUM)
    price_index: Mapped[Decimal | None] = mapped_column("PI", NUM)
    market_value: Mapped[Decimal | None] = mapped_column("MV", NUM)

    spec: Mapped[EquityIndexSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_equity_index_date_uid", "date", "uid"),)


class BondIndexData(ObservationBase):
    __tablename__ = "bond_index"

    uid: Mapped[int] = _data_uid("bond_index_spec")

    modified_duration: Mapped[Decimal | None] = mapped_column("DM", NUM)
    return_index: Mapped[Decimal | None] = mapped_column("RI", NUM)
    redemption_yield: Mapped[Decimal | None] = mapped_column("RY", NUM)
    convexity: Mapped[Decimal | None] = mapped_column("CX", NUM)
    index_level: Mapped[Decimal | None] = mapped_column("IN_", NUM)
    yield_to_worst: Mapped[Decimal | None] = mapped_column("YTW", NUM)

    spec: Mapped[BondIndexSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_bond_index_date_uid", "date", "uid"),)


class CommodityIndexData(ObservationBase):
    __tablename__ = "commodity_index"

    uid: Mapped[int] = _data_uid("commodity_index_spec")

    return_index: Mapped[Decimal | None] = mapped_column("X", NUM)

    spec: Mapped[CommodityIndexSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_commodity_index_date_uid", "date", "uid"),)


class FXRateData(ObservationBase):
    __tablename__ = "fx_rates"

    uid: Mapped[int] = _data_uid("fx_rates_spec")
    pricing_location: Mapped[str] = mapped_column(String(6), primary_key=True)

    bid: Mapped[Decimal | None] = mapped_column("EB", NUM)
    rate: Mapped[Decimal | None] = mapped_column("ER", NUM)
    open_rate: Mapped[Decimal | None] = mapped_column("EO", NUM)
    value: Mapped[Decimal | None] = mapped_column("X", NUM)

    spec: Mapped[FXRateSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_fx_rates_date_uid", "date", "uid"),)


class YieldCurveData(ObservationBase):
    __tablename__ = "yield_curve"

    uid: Mapped[int] = _data_uid("yield_curve_spec")
    tenor: Mapped[Decimal] = mapped_column(Numeric(12, 6), primary_key=True)

    redemption_yield: Mapped[Decimal | None] = mapped_column("RY", NUM)

    spec: Mapped[YieldCurveSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_yield_curve_date_uid", "date", "uid"),)


class InterestRateData(ObservationBase):
    __tablename__ = "interest_rate"

    uid: Mapped[int] = _data_uid("interest_rate_spec")

    bid: Mapped[Decimal | None] = mapped_column("IB", NUM)
    return_index: Mapped[Decimal | None] = mapped_column("RI", NUM)
    rate: Mapped[Decimal | None] = mapped_column("IR", NUM)
    open_rate: Mapped[Decimal | None] = mapped_column("IO", NUM)
    value: Mapped[Decimal | None] = mapped_column("X", NUM)

    spec: Mapped[InterestRateSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_interest_rate_date_uid", "date", "uid"),)


class HedgeFundIndexData(ObservationBase):
    __tablename__ = "hedge_fund_index"

    uid: Mapped[int] = _data_uid("hedge_fund_index_spec")

    return_index: Mapped[Decimal | None] = mapped_column("X", NUM)

    spec: Mapped[HedgeFundIndexSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_hedge_fund_index_date_uid", "date", "uid"),)


class FutureData(ObservationBase):
    __tablename__ = "future"

    uid: Mapped[int] = _data_uid("future_spec")

    last: Mapped[Decimal | None] = mapped_column("L", NUM)
    open_interest: Mapped[Decimal | None] = mapped_column("OI", NUM)
    high: Mapped[Decimal | None] = mapped_column("PH", NUM)
    low: Mapped[Decimal | None] = mapped_column("PL", NUM)
    open_price: Mapped[Decimal | None] = mapped_column("PO", NUM)
    settle: Mapped[Decimal | None] = mapped_column("PS", NUM)
    volume: Mapped[Decimal | None] = mapped_column("VM", NUM)

    spec: Mapped[FutureSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_future_date_uid", "date", "uid"),)


class FactorData(ObservationBase):
    __tablename__ = "factor"

    uid: Mapped[int] = _data_uid("factor_spec")

    excess_return: Mapped[Decimal | None] = mapped_column("XR", NUM)

    spec: Mapped[FactorSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_factor_date_uid", "date", "uid"),)


class EconomicData(ObservationBase):
    __tablename__ = "economic"

    uid: Mapped[int] = _data_uid("economic_spec")
    release_date: Mapped[dt.date] = mapped_column(Date, primary_key=True)

    value: Mapped[Decimal | None] = mapped_column("X", NUM)

    spec: Mapped[EconomicSpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_economic_date_uid_release", "date", "uid", "release_date"),)


class ImpliedVolatilityData(ObservationBase):
    __tablename__ = "implied_volatility"

    uid: Mapped[int] = _data_uid("implied_volatility_spec")
    pricing_location: Mapped[str] = mapped_column(String(6), primary_key=True)
    tenor: Mapped[str] = mapped_column(String(16), primary_key=True)
    strike: Mapped[str] = mapped_column(String(40), primary_key=True)

    strike_reference: Mapped[str | None] = mapped_column(String(20))
    relative_strike: Mapped[str | None] = mapped_column(String(20))

    bid: Mapped[Decimal | None] = mapped_column(NUM)
    mid: Mapped[Decimal | None] = mapped_column(NUM)
    ask: Mapped[Decimal | None] = mapped_column(NUM)

    spec: Mapped[ImpliedVolatilitySpec] = relationship(back_populates="data")

    __table_args__ = (Index("ix_implied_volatility_date_uid", "date", "uid"),)


# =============================================================================
# Registries and query helpers
# =============================================================================


SPEC_MODELS: tuple[type[TimeSeriesSpec], ...] = (
    EquitySpec,
    ETFSpec,
    EquityIndexSpec,
    BondIndexSpec,
    CommodityIndexSpec,
    FXRateSpec,
    YieldCurveSpec,
    InterestRateSpec,
    HedgeFundIndexSpec,
    FutureSpec,
    FactorSpec,
    EconomicSpec,
    ImpliedVolatilitySpec,
)


TS_MODEL_BY_TYPE: dict[str, type[ObservationBase]] = {
    "equity": EquityData,
    "etf": ETFData,
    "equity_index": EquityIndexData,
    "bond_index": BondIndexData,
    "commodity_index": CommodityIndexData,
    "fx_rate": FXRateData,
    "yield_curve": YieldCurveData,
    "interest_rate": InterestRateData,
    "hedge_fund_index": HedgeFundIndexData,
    "future": FutureData,
    "factor": FactorData,
    "economic": EconomicData,
    "implied_volatility": ImpliedVolatilityData,
}


SPEC_MODEL_BY_TYPE: dict[str, type[TimeSeriesSpec]] = {
    model.__mapper_args__["polymorphic_identity"]: model for model in SPEC_MODELS
}


def timeseries_model_for(spec: TimeSeriesSpec) -> type[ObservationBase]:
    try:
        return TS_MODEL_BY_TYPE[spec.asset_type]
    except KeyError as exc:
        raise LookupError(f"No observation model registered for asset_type={spec.asset_type!r}") from exc


def mapped_columns_as_dict(obj: object) -> dict[str, Any]:
    mapper = inspect(obj).mapper
    return {attr.key: getattr(obj, attr.key) for attr in mapper.column_attrs}


def get_specs_by_ticker(session: Session, ticker: str) -> list[TimeSeriesSpec]:
    """Query 1: base spec columns plus subtype columns, loaded polymorphically."""

    stmt = (
        select(TimeSeriesSpec)
        .where(TimeSeriesSpec.ticker == ticker)
        .options(selectin_polymorphic(TimeSeriesSpec, list(SPEC_MODELS)))
        .order_by(TimeSeriesSpec.uid)
    )
    return list(session.scalars(stmt))


def get_specs_by_ticker_one_join(session: Session, ticker: str) -> list[TimeSeriesSpec]:
    """Alternative Query 1 for small lookups: one left-joined polymorphic selectable."""

    spec_poly = with_polymorphic(TimeSeriesSpec, list(SPEC_MODELS))
    stmt = select(spec_poly).where(spec_poly.ticker == ticker).order_by(spec_poly.uid)
    return list(session.scalars(stmt))


def get_timeseries_by_uid(
    session: Session,
    uid: int,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[ObservationBase]:
    """Query 2: resolve uid -> spec -> concrete observation table via registry."""

    spec = session.get(TimeSeriesSpec, uid)
    if spec is None:
        return []

    model = timeseries_model_for(spec)
    stmt = select(model).where(model.uid == uid).order_by(model.date)

    if start is not None:
        stmt = stmt.where(model.date >= start)
    if end is not None:
        stmt = stmt.where(model.date <= end)

    return list(session.scalars(stmt))


def get_timeseries_dicts_by_uid(
    session: Session,
    uid: int,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    return [mapped_columns_as_dict(row) for row in get_timeseries_by_uid(session, uid, start, end)]
