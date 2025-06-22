from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CPrivateCredit']

class CPrivateCredit(CPrivateAsset):
    _asset_name = 'PRIVATE_CREDIT'
    _pme_name = 'LHYIELD_GE20'

    _reporting_name = 'Private Credit'
    _category = 'Other Private Assets'

    def __init__(self, schema):
        CPrivateAsset.__init__(
            self,
            schema,
            CPrivateCredit._pme_name,
            CPrivateCredit._asset_name)

if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CPrivateCredit(schema)
    rp_pme = tt.get_risk_premias()