from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional

__all__ = []


class CLendingRate(CAsset):

    def __init__(
            self,
            name: str,
            schema: CContext,
            spread: float,
            borrowing_cost,
    ) -> None:
        # the base rate is the schema currency risk free rate
        rfr = schema.get_risk_free_rate_asset()
        super(CLendingRate, self).__init__(
            rfr,
            schema,
            **rfr.get_asset_params(),
        )

        self._spread = spread
        self._borrowing_cost = borrowing_cost
        self.name = name

        self.set_reporting_info(
            name,
            'Fixed Income'
        )

    def get_reference_rate_time_seres(self):
        pass

    def get_spread(self):
        pass

    def get_borrowing_cost(self):
        pass

    def get_risk_premia(self):
        return 0

    def get_risk_premias(self):
        return super(CLendingRate, self).get_risk_premias() * 0


if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    rate = CLendingRate(
        "Libor100",
        schema,
        spread=0.005,
        borrowing_cost=0.01
    )
