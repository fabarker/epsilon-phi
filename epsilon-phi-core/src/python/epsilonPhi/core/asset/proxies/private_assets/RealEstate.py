from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CRealEstate']

class CRealEstate(CPrivateAsset):
    _asset_name = 'PA_REAL_ESTATE'
    _pme_name = 'INFRA_EQUITY'

    _reporting_name = 'Private Real Estate'
    _category = 'Other Private Assets'

    def __init__(self, schema):
        CPrivateAsset.__init__(
            self,
            schema,
            CRealEstate._pme_name,
            CRealEstate._asset_name)

if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CRealEstate(schema)
    rp_pme = tt.get_risk_premias()