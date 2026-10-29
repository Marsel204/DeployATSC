"""Fresh copies of owned JSON-like evidence; other Python values retain deepcopy semantics."""
from copy import deepcopy


_ATOMIC=(str,int,float,bool,type(None))


def clone(value):
    try:return _clone(value)
    except RecursionError:return deepcopy(value)


def _clone(value):
    kind=type(value)
    if kind in _ATOMIC:return value
    if kind is dict:return {key if type(key) is str else deepcopy(key):_clone(item) for key,item in value.items()}
    if kind is list:return [_clone(item) for item in value]
    if kind is tuple:return tuple(_clone(item) for item in value)
    return deepcopy(value)
