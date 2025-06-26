from .BankLoans import *
from .HighYield import *
from .IncomeEquity import *
from .Infra import *
from .LendingRate import *
from .private_assets import *
from .SingleStock import *
from .TIPs import *
from .hedge_funds import *
from .private_assets.custom_pmes import *

from functools import lru_cache
import inspect


@lru_cache(maxsize=None)
def get_proxy_object_from_name(target_name: str):
    for obj in globals().values():
        if inspect.isclass(obj):
            asset_name = getattr(obj, "_asset_name", None)
            if asset_name == target_name:
                return obj
    raise ValueError(f"No matching class found with _ASSET_NAME = '{target_name}'")


__all__ = (
    'get_class_object_from_name'
)
