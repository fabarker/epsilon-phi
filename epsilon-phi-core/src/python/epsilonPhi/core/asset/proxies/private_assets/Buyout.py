from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CBuyout']


class CBuyout(CPrivateAsset):
    _asset_name = 'PE_BUYOUT'
    _pme_name = 'MSWRLD$'

    _reporting_name = 'Buyout'
    _category = 'Private Equity'

    def __init__(self, schema, **kwargs):
        pme = kwargs.pop('data', schema.get_asset_from_name(CBuyout._pme_name))
        CPrivateAsset.__init__(self, pme, schema, CBuyout._asset_name, **kwargs)
        self.set_reporting_info(CBuyout._reporting_name, CBuyout._category)

if __name__ == "__main__":
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CBuyout(schema)
    tt.add(1)
    lvls = tt.get_levels()
