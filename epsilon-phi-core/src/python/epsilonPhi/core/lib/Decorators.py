import datetime
from functools import wraps
from enum import Enum
from sqlalchemy.util import NoneType


class SingletonDecorator:
    def __init__(self, klass, **kwargs):
        self.klass = klass
        self.instance = None
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'instance') or self.instance is None:
            self.instance = self.klass(*args, **dict(self.kwargs, **kwargs))
        return self.instance

    def cleanup(self):
        self.instance = None


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


builtin_types = [datetime.datetime, datetime.date, NoneType, Enum, int, float, bool, str, enumerate, bytes]


def auto_repr(cls):
    @wraps(cls)
    def __repr__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % (item, str(getattr(self, item))) for item in dir(self) if
                      not callable(item) and not item.startswith("_") and
                      type(getattr(self, item)) in builtin_types)
        )

    cls.__repr__ = __repr__
    return cls
