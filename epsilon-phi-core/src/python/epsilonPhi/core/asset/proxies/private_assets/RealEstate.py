from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CRealEstate']


class CRealEstate(CPrivateAsset):
    _asset_name = 'PA_REAL_ESTATE'
    _pme_name = 'REAL_ESTATE_PME'

    _reporting_name = 'Private Real Estate'
    _category = 'Other Private Assets'

    def __init__(self, schema, **kwargs):
        pme = kwargs.pop('data', schema.get_asset_from_name(CRealEstate._pme_name))
        CPrivateAsset.__init__(self, pme, schema, CRealEstate._asset_name)
        self.set_reporting_info(CRealEstate._reporting_name, CRealEstate._category)


if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CRealEstate(schema)
    rp_pme = tt.get_risk_premias()
