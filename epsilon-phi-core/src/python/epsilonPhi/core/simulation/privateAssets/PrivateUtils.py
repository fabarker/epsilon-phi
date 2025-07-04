from epsilonPhi.core.simulation.privateAssets.vintage import Vintage
import numpy as np
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset

class CPrivateProgram(object):
    pass


class CPrivateUtils(object):

    def __init__(self, schema):

        self._schema = schema
        self._capital_calls_and_distr_assumptions = {}
        self._cash_flow_defaults = {}

    # GeneratePECashFlowSub
    def get_cash_flows(self,
                       current_total_liquid_assets,
                       inital_vintages,
                       annual_commitments: np.array,
                       inflation_paths=None,
                       ann_commitments_in_dollars=False,
                       wealth_flows=None,
                       num_years=20,
                       use_total_mv=False,
                       ):

        # VintageYear(commitment: float = 0, initial_value = 0, start_age= 0, calls, distributions, returns, shocks)
        # new_vintages = np.tile(Vintage(0, 0, 0), [len(inital_vintages), num_years])

        num_classes = annual_commitments.shape[0]
        distributions = np.zeros([num_classes, num_years])
        capital_calls = np.zeros([num_classes, num_years])
        pe_alloc = np.zeros([num_classes, num_years])
        vintage_exposures = np.zeros([num_years, num_years, num_classes])
        liquid_assets_EOY = np.zeros(num_years)
        total_MV = np.zeros(num_years)

        # loop through number of years
        for year in range(num_years):

            # Loop through all vintages and compute distributions from previous yaer
            for i, asset in enumerate(inital_vintages):

                # for the vintages we have, get the distributions in year
                distributions[i, year] = asset.get_distribution(year)

                for fund_start_year in range(year + 1):
                    distributions[i, year] += new_vintages[i, fund_start_year].distribution(year - fund_start_year)

            # Compute the total market value at beginning of the year
            for i, asset in enumerate(inital_vintages):
                pe_alloc[i, year] = sum([vy.mv_start_of_year(year) for vy in asset])
                vintage_exposures[0, year, i] = sum([vy.mv_start_of_year(year) for vy in asset])

                for fund_start_year in range(year + 1):
                    pe_alloc[i, year] += new_vintages[i, fund_start_year].mv_start_of_year(year - fund_start_year)
                    vintage_exposures[fund_start_year, year, i] =  new_vintages[i, fund_start_year].mv_start_of_year(year - fund_start_year)

            total_MV[year] = current_total_liquid_assets if year == 0 else liquid_assets_EOY[year-1] * sum(pe_alloc[:, year])

            # Apply Inflows and Outflows at the end of the year
            if wealth_flows is not None:
               pass

            # Now grow the liquid assets given their return. The after growth NAV will take care of the flows, recieving distributions
            liquid_assets_EOY_pre_flows = current_total_liquid_assets if year == 0 else liquid_assets_EOY[year-1] * (1 + returns[year])
            liquid_assets_EOY[year] = liquid_assets_EOY_pre_flows + np.sum(distributions[:, year]) + flows[year]

            # now we need to create new commitments
            for i, asset in enumerate(inital_vintages):
                if year == 0:
                    rtns = returns["private"][i]
                else:
                    rtns = returns["private"][i][year:]
                    for i in range(year):
                        rtns = np.append(rtns, [0.0])

                if not ann_commitments_in_dollars:
                    ann_commitments_in_dollars[i, year] = annual_commitments[i, year] * (total_MV[year] if use_total_mv else liquid_assets_EOY[year])

                new_vintages[i, year] = Vintage(
                    ann_commitments_in_dollars[i, year],
                    0,
                    0,
                )

            for i, asset in enumerate(inital_vintages):
                capital_calls[i, year] = sum([vy.call_ammount(year) for vy in asset])

                for fund_start_year in range(year + 1):
                    capital_calls[i, year] += new_vintages[i, fund_start_year].call_ammount(year - fund_start_year)

                # subtract capital calls from liquid assets
                liquid_assets_EOY[year] -= capital_calls[i, year]

        pe_percent = pe_alloc / total_MV
        return capital_calls, distributions, pe_percent, ann_commitments_in_dollars, total_MV, flows, vintage_exposures


if __name__ == "__main__":


    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    current_total_liquid_assets = 100
    annual_commitments = np.ones((1, 20)) * 10
    initial_vintages = [
        Vintage(PrivateAsset.BUYOUT, schema=schema, commitment_size=20, fund_age=1),
        Vintage(PrivateAsset.BUYOUT, schema=schema, commitment_size=4, fund_age=2),
        Vintage(PrivateAsset.BUYOUT, schema=schema, commitment_size=10, fund_age=6)
    ]

    utils = CPrivateUtils(schema)

    utils.get_cash_flows(
        current_total_liquid_assets,
        initial_vintages,
        annual_commitments,
        inflation_paths = None,
        ann_commitments_in_dollars = True,
        wealth_flows = None,
        num_years = 20,
        use_total_mv = False,
    )

