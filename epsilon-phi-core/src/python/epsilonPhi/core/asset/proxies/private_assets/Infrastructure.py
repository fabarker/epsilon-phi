from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CInfrastructure']

class CInfrastructure(CPrivateAsset):
    _asset_name = 'PA_INFRA'
    _pme_name = 'INFRA_EQUITY'

    _reporting_name = 'Infrastructure'
    _category = 'Other Private Assets'

    def __init__(self, schema):
        CPrivateAsset.__init__(
            self,
            schema,
            CInfrastructure._pme_name,
            CInfrastructure._asset_name)


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CInfrastructure(schema)
    rp_pme = tt.get_risk_premias()