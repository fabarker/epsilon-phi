from datetime import datetime, timedelta
from epsilonPhi.core.utils.DateUtils import DateUtils
import numpy as np
import QuantLib as ql
import pandas as pd

class BondUtils(object):
    pass

    @staticmethod
    def price(issue_date, pricing_date, maturity_at_issue, coupon_rate, yield_tm, coupon_frequency,
                   day_count_convention="30/360", price_type='dirty'):

        # Convert string dates to datetime objects
        ID = pd.to_datetime(issue_date).to_pydatetime()
        PD = pd.to_datetime(pricing_date).to_pydatetime()
        MD = issue_date + timedelta(days=DateUtils.days_per_year * maturity_at_issue)

        issue_date = ql.Date(ID.day, ID.month, ID.year)
        maturity_date = ql.Date(MD.day, MD.month, MD.year)
        settlement_date = ql.Date(PD.day, PD.month, PD.year)

        if day_count_convention == "30/360":
            day_count = ql.Thirty360(ql.Thirty360.USA)
        elif day_count_convention == "30/365":
            day_count = ql.Thirty365()
        elif day_count_convention == "Actual/365":
            day_count = ql.Actual36525()
        elif day_count_convention == "Actual/360":
            day_count = ql.Actual360()
        elif day_count_convention == "Actual/Actual":
            day_count = ql.ActualActual(ql.ActualActual.ISMA)
        else:
            raise ValueError("Unsupported day count convention")

        schedule = schedule = ql.Schedule(issue_date,
                                          maturity_date,
                                          ql.Period(coupon_frequency),
                                          ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                                          ql.Following,
                                          ql.Following,
                                          ql.DateGeneration.Backward,
                                          False)

        if DateUtils.is_iterable(coupon_rate):
           coupon_rate = coupon_rate[0]

        if DateUtils.is_iterable(yield_tm):
           yield_tm = yield_tm[0]

        bond = ql.FixedRateBond(0, 100, schedule, [coupon_rate], day_count, ql.Following, 100, issue_date)
        if price_type.lower() == 'dirty':
            return bond.dirtyPrice(yield_tm, day_count, ql.SimpleThenCompounded, coupon_frequency, settlement_date)
        elif price_type.lower() == 'clean':
            return bond.cleanPrice(yield_tm, day_count, ql.SimpleThenCompounded, coupon_frequency, settlement_date)


    @staticmethod
    def convertYield(df_, maturity, coupon_frequency=2, basis="Actual/Actual"):

        prices = list()
        for PD, row in df_.iterrows():
            if PD > df_.index.min():
               price = BondUtils.price(issue_date=df_.index[df_.index < PD].max(),
                                               pricing_date=PD,
                                               maturity_at_issue=maturity,
                                               coupon_rate=df_.loc[df_.index[df_.index < PD].max()].values / 100,
                                               yield_tm=row.values / 100,
                                               coupon_frequency=coupon_frequency,
                                               day_count_convention=basis)
            else:
                price = np.nan
            prices.extend([price])
        df = pd.DataFrame(prices, index=df_.index)
        return -1 + (df/100)

if __name__ == "__main__":

    tickers = ['TRUK1MT', 'TRUK3MT']
    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()

    df_ = pd.DataFrame()
    for ticker in tickers:
        df = gds.get_dataframe_from_ticker(ticker)
        df_ = pd.concat((df, df_), axis=1)

    df_M = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='BM'))
    returns = BondUtils.convertYield(df.get(ticker).get('RY').to_frame('RY'), maturity=10)
