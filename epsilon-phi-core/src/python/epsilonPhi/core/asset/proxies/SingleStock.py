from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional

class CSingleStock(CAsset):

    def __init__(
            self,
            schema: Optional[CContext] = None,
            **kwargs
    ) -> None:

        data = self.get_time_series()
        super().__init__(data, schema, **kwargs)
