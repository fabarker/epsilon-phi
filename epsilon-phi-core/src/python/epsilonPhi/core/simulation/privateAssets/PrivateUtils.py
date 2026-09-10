from epsilonPhi.core.simulation.privateAssets.vintage import Vintage
import numpy as np
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
from epsilonPhi.core.simulation.SimStructs import PrivateAssetProjections
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency

class CPrivateProgram(object):
    pass


class CPrivateUtils(object):

    def __init__(self, schema):

        self._schema = schema
        self._capital_calls_and_distr_assumptions = {}
        self._cash_flow_defaults = {}

    @staticmethod
    def get_liquid_portfolio_yearly_returns(ptf_manager):
        liq_ptf = CPrivateUtils.get_public_portfolio(ptf_manager.portfolio)
        liquid_portfolio_yearly_returns, _ = liq_ptf.get_portfolio_simulated_returns_panel(frequency=Frequency.YEARLY)
        return np.mean(liquid_portfolio_yearly_returns, axis=1)

    @staticmethod
    def get_public_portfolio(ptf):

        priv_assets = [x for x in ptf.get_asset_names() if "pri" in ptf.get_asset(x).category.lower()]
        if len(ptf.get_asset_names()) > len(priv_assets):
            return ptf.remove_assets(priv_assets, True)
        else:
            raise ValueError("No public assets in portfolio")

    @staticmethod
    def get_private_asset_weights(ptf):

        # Get private assets list
        priv_assets = [x for x in ptf.get_asset_names() if "pri" in ptf.get_asset(x).category.lower()]

        # Get liquid assets list
        liq_assets = np.setdiff1d(ptf.get_asset_names(), priv_assets)

        return ptf.remove_assets(liq_assets, False)

    @staticmethod
    def split_portfolio_into_liquid_and_private_assets(ptf):

        # Get private assets list
        priv_assets = [x for x in ptf.get_asset_names() if "pri" in ptf.get_asset(x).category.lower()]

        # Get liquid assets list
        liq_assets = np.setdiff1d(ptf.get_asset_names(), priv_assets)

        if len(priv_assets) > 0 and len(liq_assets) > 0:
            prv_ptf = ptf.remove_assets_and_rebalance(liq_assets) if len(priv_assets)  else ptf.deepcopy()
            pub_ptf = ptf.remove_assets_and_rebalance(priv_assets) if len(priv_assets) else ptf.deepcopy()
            return prv_ptf, pub_ptf
        else:
            raise ValueError("Error in spliting public and private portfolios")

    def generate_commitments(self, strategies, years, system=None):
       #if system == "PMG":
       # Overide frequencies for secondaries to every third year [0, 0, 1]
       #pass
       commits =  np.ones((len(strategies), years))
       return commits

    def generate_private_markets_cash_flows(
            self,
            initial_vintages,
            annual_commitments: np.ndarray,
            num_years: int = 20
    ) -> PrivateAssetProjections:

        """
        Simulates PE cash flows over time, including capital calls, distributions, and NAVs.
        The function is agnostic to liquid assets and only consideres the private markets portfolio

        Args:
            initial_vintages: A dictionary mapping strategy/asset names to a list of Vintage objects.
            annual_commitments: A [num_assets x num_years] array of dollar commitments.
            num_years: Number of years to simulate.

        Returns:
            PrivateAssetProjections: containing capital calls, distributions, NAVs, and exposures.
        """

        num_classes = annual_commitments.shape[0]
        assert num_classes == len(initial_vintages), "Error - mismatch between vintages and commitment arrays"
        distributions = np.zeros([num_classes, num_years])
        capital_calls = np.zeros([num_classes, num_years])
        pe_alloc = np.zeros([num_classes, num_years])
        vintage_exposures = np.zeros([num_years, num_years, num_classes])
        total_MV = np.zeros(num_years)

        new_vintages = [
            [Vintage(x, self._schema, 0) for _ in range(num_years)]
            for x in initial_vintages.keys()
        ]

        # loop through number of years
        for year in range(num_years):

            # Loop through all vintages and compute distributions from previous yaer
            for i, asset in enumerate(initial_vintages):

                # for the vintages we have, get the distributions in year
                distributions[i, year] = np.sum([vy.get_distribution(year) for vy in initial_vintages.get(asset)])

                for fund_start_year in range(year + 1):
                    distributions[i, year] += new_vintages[i][fund_start_year].get_distribution(year - fund_start_year)

            # Compute the total market value at beginning of the year
            for i, asset in enumerate(initial_vintages):
                pe_alloc[i, year] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])
                vintage_exposures[0, year, i] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])

                for fund_start_year in range(year + 1):
                    pe_alloc[i, year] += new_vintages[i][fund_start_year].get_BOY_NAV(year - fund_start_year)
                    vintage_exposures[fund_start_year, year, i] = new_vintages[i][fund_start_year].get_BOY_NAV(
                        year - fund_start_year)

            total_MV[year] = sum(pe_alloc[:, year])

            # now we need to create new commitments
            for i, asset in enumerate(initial_vintages):

                # construct new vintage year
                new_vintages[i][year] = Vintage(
                    new_vintages[i][year].type,
                    self._schema,
                    annual_commitments[i, year],
                )

            # get total capital calls by summing across all vintages
            for i, asset in enumerate(initial_vintages):
                capital_calls[i, year] = np.sum([vy.get_capital_call(year) for vy in initial_vintages.get(asset)])

                for fund_start_year in range(year + 1):
                    capital_calls[i, year] += new_vintages[i][fund_start_year].get_capital_call(year - fund_start_year)

        return PrivateAssetProjections(
            capital_calls,
            distributions,
            initial_vintages.keys(),
            total_MV, pe_alloc,
            annual_commitments,
            vintage_exposures
        )



    # GeneratePECashFlowSub
    def generate_cash_flows_liquid_private(self,
                               current_total_liquid_assets,
                               initial_vintages,
                               annual_commitments: np.array,
                               liquid_portfolio_yearly_returns,
                               inflation_paths=None,
                               ann_commitments_in_dollars=False,
                               wealth_flows=None,
                               num_years=20,
                               use_total_mv=False,
                               ):

        # VintageYear(commitment: float = 0, initial_value = 0, start_age= 0, calls, distributions, returns, shocks)
        # new_vintages = np.tile(Vintage(0, 0, 0), [len(inital_vintages), num_years])

        num_classes = annual_commitments.shape[0]
        assert num_classes == len(initial_vintages), "Error - mismatch between vintages and commitment arrays"
        distributions = np.zeros([num_classes, num_years])
        capital_calls = np.zeros([num_classes, num_years])
        pe_alloc = np.zeros([num_classes, num_years])
        vintage_exposures = np.zeros([num_years, num_years, num_classes])
        liquid_assets_EOY = np.zeros(num_years)
        total_MV = np.zeros(num_years)


        new_vintages = [
            [Vintage(x, self._schema, 0) for _ in range(num_years)]
            for x in initial_vintages.keys()
        ]

        # loop through number of years
        for year in range(num_years):

            # Loop through all vintages and compute distributions from previous yaer
            for i, asset in enumerate(initial_vintages):

                # for the vintages we have, get the distributions in year
                distributions[i, year] = np.sum([ vy.get_distribution(year) for vy in initial_vintages.get(asset) ])

                for fund_start_year in range(year + 1):
                    distributions[i, year] += new_vintages[i][fund_start_year].get_distribution(year - fund_start_year)

            # Compute the total market value at beginning of the year
            for i, asset in enumerate(initial_vintages):
                pe_alloc[i, year] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])
                vintage_exposures[0, year, i] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])

                for fund_start_year in range(year + 1):
                    pe_alloc[i, year] += new_vintages[i][fund_start_year].get_BOY_NAV(year - fund_start_year)
                    vintage_exposures[fund_start_year, year, i] =  new_vintages[i][fund_start_year].get_BOY_NAV(year - fund_start_year)

            total_MV[year] = (current_total_liquid_assets if year == 0 else liquid_assets_EOY[year-1]) + sum(pe_alloc[:, year])

            # Apply Inflows and Outflows at the end of the year
            if wealth_flows is not None:
               pass

            flows = np.zeros((20, ))

            # Now grow the liquid assets given their return. The after growth NAV will take care of the flows, recieving distributions
            # On year 0, use initial capital; otherwise, grow previous year’s liquid assets
            liquid_assets_EOY_pre_flows = (
                current_total_liquid_assets
                if year == 0
                else liquid_assets_EOY[year - 1] * (1 + liquid_portfolio_yearly_returns[year])
            )

            liquid_assets_EOY[year] = liquid_assets_EOY_pre_flows + np.sum(distributions[:, year]) + flows[year]

            # now we need to create new commitments
            for i, asset in enumerate(initial_vintages):

                # Determine the new vintage commitment amount
                vintage_commit = (
                    annual_commitments[i, year]
                    if ann_commitments_in_dollars
                    else annual_commitments[i, year] * (
                        total_MV[year] if use_total_mv else liquid_assets_EOY[year]
                    )
                )

                # construct new vintage year
                new_vintages[i][year] = Vintage(
                    new_vintages[i][year].type,
                    self._schema,
                    vintage_commit,
                )

            # get total capital calls by summing across all vintages
            for i, asset in enumerate(initial_vintages):
                capital_calls[i, year] = np.sum([vy.get_capital_call(year) for vy in initial_vintages.get(asset)])

                for fund_start_year in range(year + 1):
                    capital_calls[i, year] += new_vintages[i][fund_start_year].get_capital_call(year - fund_start_year)

                # subtract capital calls from liquid assets
                liquid_assets_EOY[year] -= capital_calls[i, year]

        return PrivateAssetProjections(
            capital_calls,
            distributions,
            initial_vintages.keys(),
            total_MV, pe_alloc,
            annual_commitments,
            vintage_exposures
        )

    def multiplier(self, wts_tgt, wts_cur, mult, year):

        assert wts_tgt.shape[0] == wts_cur.shape[0],  ValueError("wts_tgt.shape != wts_cur.shape")
        max_mult = mult * wts_tgt.sum()
        wts_tgt_tmp = wts_tgt.copy()
        wts_tgt_tmp[wts_tgt_tmp == 0] = np.finfo(float).eps

        rlt_dist_tgt = 1 - np.minimum(wts_cur[:, year] / wts_tgt_tmp.flat, 1)
        return 1 + max_mult * rlt_dist_tgt


    def vintage_dicts_to_objects(self, existing_vintages):

        # This function takes a list of dictionaries that contain vintage information and
        # translates into vintage objects

        vintage_dict = {}
        for idx, strategy in enumerate(existing_vintages.keys()):

            vintages_of_this_strategy = existing_vintages.get(strategy)
            if len(vintages_of_this_strategy) > 0:

                # loop through each vintage
                strat_vintages = []
                for v in vintages_of_this_strategy:
                    curr_vin_year = Vintage(
                        PrivateAsset.get(strategy),
                        self._schema,
                        commitment_size=v.get("commitment_size"),
                        realized_nav=v.get("realized_nav", 0),
                        fund_age=v.get("fund_age", 0),
                        cumltv_realized_contributions=v.get("cumltv_realized_contributions", 0),
                        cumltv_realized_distributions=v.get("cumltv_realized_distributions", 0))

                    # Dump the vintage strategy here
                    strat_vintages.append(curr_vin_year)
                vintage_dict[PrivateAsset.get(strategy)] = strat_vintages
            else:
                curr_vin_year = Vintage(
                    PrivateAsset.get(strategy),
                    self._schema,
                    commitment_size=0)
                vintage_dict[PrivateAsset.get(strategy)] = [curr_vin_year]
        return vintage_dict







if __name__ == "__main__":


    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    initial_vintage = {
        "PE_BUYOUT":
            [
                {
                    'fund_age': 0,
                    'commitment_size': 100,
                    'realized_nav': 0,
                    'cumltv_realized_contributions': 0,
                    'cumltv_realized_distributions': 0
                }
            ],
        "PRIVATE_CREDIT":
            [
                {
                    'fund_age': 0,
                    'commitment_size': 100,
                    'realized_nav': 0,
                    'cumltv_realized_contributions': 0,
                    'cumltv_realized_distributions': 0
                }
            ],
    }

    utils = CPrivateUtils(schema)

    annual_commitments = np.ones((2, 20)) * 0

    init_vint = utils.vintage_dicts_to_objects(initial_vintage)
    res = utils.generate_private_markets_cash_flows(init_vint, annual_commitments)

