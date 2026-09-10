from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CGrowth']


class CGrowth(CPrivateAsset):
    _asset_name = 'PE_GROWTH'
    _pme_name = 'GROWTH_PME'

    _reporting_name = 'Growth'
    _category = 'Private Equity'

    def __init__(self, schema, **kwargs):
        pme = kwargs.pop('data', schema.get_asset_from_name(CGrowth._pme_name))
        CPrivateAsset.__init__(self, pme, schema, CGrowth._asset_name)
        self.set_reporting_info(CGrowth._reporting_name, CGrowth._category)


if __name__ == "__main__":
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CGrowth(schema)
    rp_pme = tt.get_risk_premias()
