from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CBuyout']

class CBuyout(CPrivateAsset):
    _asset_name = 'PE_BUYOUT'
    _pme_name = 'MSWRLD$'

    def __init__(self, schema):
        CPrivateAsset.__init__(
            self,
            schema,
            CBuyout._pme_name,
            CBuyout._asset_name)

if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CBuyout(schema)
    rp_pme = tt.get_risk_premias()