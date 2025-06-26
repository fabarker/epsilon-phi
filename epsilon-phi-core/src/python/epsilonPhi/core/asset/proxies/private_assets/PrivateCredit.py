from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CPrivateCredit']


class CPrivateCredit(CPrivateAsset):
    _asset_name = 'PRIVATE_CREDIT'
    _pme_name = 'LHYIELD_GE20'

    _reporting_name = 'Private Credit'
    _category = 'Other Private Assets'

    def __init__(self, schema, **kwargs):
        pme = kwargs.pop('data', schema.get_asset_from_name(CPrivateCredit._pme_name))
        CPrivateAsset.__init__(self, pme, schema, CPrivateCredit._asset_name)
        self.set_reporting_info(CPrivateCredit._reporting_name, CPrivateCredit._category)


if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CPrivateCredit(schema)
    rp_pme = tt.get_risk_premias()
