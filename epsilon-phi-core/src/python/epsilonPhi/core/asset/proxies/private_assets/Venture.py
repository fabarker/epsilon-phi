from epsilonPhi.core.asset.proxies.private_assets.PrivateAssets import CPrivateAsset

__all__ = ['CVenture']


class CVenture(CPrivateAsset):
    _asset_name = 'PE_VENTURE'
    _pme_name = 'VENTURE_PME'

    _reporting_name = 'Venture'
    _category = 'Private Equity'

    def __init__(self, schema, **kwargs):
        pme = kwargs.pop('data', schema.get_asset_from_name(CVenture._pme_name))
        CPrivateAsset.__init__(self, pme, schema, CVenture._asset_name)
        self.set_reporting_info(CVenture._reporting_name, CVenture._category)


if __name__ == "__main__":
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CVenture(schema)
    rp_pme = tt.get_risk_premias()
