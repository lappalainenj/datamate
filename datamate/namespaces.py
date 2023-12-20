"""
This module exports `Namespace`, a `dict` that supports accessing items at
attributes, for convenience, and to better support static analysis.

It also exports`namespacify`, a function that recursively converts mappings and
Namespace-like containers in JSON-like objects to `Namespace`s.
"""

from typing import Mapping, Union
from copy import deepcopy
from numpy import ndarray
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import pandas as pd

__all__ = [
    "is_disjoint", 
    "is_subset", 
    "is_superset", 
    "to_dict",
    "depth",
    "pformat",
    "compare",
    "all_true",
    "dict_walk",
    "diff", 
    "without",
]


def is_subset(dict1, dict2):
    """
    Check whether dict2 is a subset of dict1.

    Parameters:
    dict1 (dict): The superset dictionary.
    dict2 (dict): The subset dictionary.

    Returns:
    bool: True if dict2 is a subset of dict1, False otherwise.
    """
    for key, value in dict2.items():
        if key not in dict1:
            return False
        if isinstance(value, dict):
            if not is_subset(dict1[key], value):
                return False
        else:
            if dict1[key] != value:
                return False
    return True


def is_superset(dict1, dict2):
    """
    Check whether dict2 is a superset of dict1.

    Parameters:
    dict1 (dict): The subset dictionary.
    dict2 (dict): The superset dictionary.

    Returns:
    bool: True if dict2 is a superset of dict1, False otherwise.
    """
    return is_subset(dict2, dict1)


def is_disjoint(dict1, dict2):
    """Check whether two dictionaries are disjoint."""
    dict1_keys = set(key for key, _ in dict_walk(dict1))
    dict2_keys = set(key for key, _ in dict_walk(dict2))
    return dict1_keys.isdisjoint(dict2_keys)


def to_dict(obj):
    if isinstance(obj, dict):
        return dict((k, to_dict(v)) for k, v in obj.items())
    elif isinstance(obj, DictConfig):
        return OmegaConf.to_object(obj)
    else:
        return obj


def to_df(obj, name="", separator="."):
    as_dict = to_dict(obj)  # namespace need deepcopy method
    df = pd.json_normalize(as_dict, sep=separator).T
    if name:
        df = df.rename({0: name}, axis=1)
    return df


def depth(cls):
    if isinstance(cls, (dict, DictConfig)):
        return 1 + (max(map(depth, cls.values())) if cls else 0)
    return 0


def pformat(cls):
    import pprint

    pretty_printer = pprint.PrettyPrinter(depth=100)
    return pretty_printer.pformat(cls)


def compare(obj1, obj2):
    """
    Type agnostic comparison (for most basic types) and nested dicts.
    """
    if isinstance(obj1, (type(None), bool, int, float, str, type)) and isinstance(
        obj2, (type(None), bool, int, float, str, type)
    ):
        return obj1 == obj2
    elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        return False if len(obj1) != len(obj2) else obj1 == obj2
    elif isinstance(obj1, (ndarray)) and isinstance(obj2, (ndarray)):
        return compare(obj1.tolist(), obj2.tolist())
    elif isinstance(obj1, Mapping) and isinstance(obj2, Mapping):
        _obj1, _obj2 = obj1.deepcopy(), obj2.deepcopy()
        out = {}
        for key in (
            set(_obj1.keys())
            .difference(set(_obj2.keys()))
            .union(set(_obj2.keys()).difference(set(_obj1.keys())))
        ):
            out[key] = False
            _obj1.pop(key, None)
            _obj2.pop(key, None)
        for k in _obj1:
            out[k] = compare(_obj1[k], obj2[k])
        return DictConfig(out)
    elif type(obj1) != type(obj2):
        return False


def all_true(obj: object) -> bool:
    """
    Return True if bool(element) is True for all elements in nested obj.
    """
    if isinstance(obj, (type(None), bool, int, float, str, type, bytes)):
        return bool(obj)
    elif isinstance(obj, Path):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return all([all_true(v) for v in obj])
    elif isinstance(obj, (ndarray)):
        return all([all_true(v.item()) for v in obj])
    elif isinstance(obj, Mapping):
        return all([all_true(obj[k]) for k in obj])
    else:
        try:
            return all_true(vars(obj))
        except TypeError as e:
            raise TypeError(f"all {obj} of type {type(obj)}: {e}.") from e


def dict_walk(dictionary):
    """
    Recursively walk through a nested dictionary and yield a tuple
    for each key-value pair.

    Parameters:
    dictionary (dict): The dictionary to walk through.

    Yields:
    tuple: A tuple containing the current key and its corresponding value.
    """
    for key, value in dictionary.items():
        yield (key, value)
        if isinstance(value, dict):
            yield from dict_walk(value)


def diff(obj1, obj2, name1="obj", name2="other"):
    """Diff two mappables."""
    if obj1 is None or obj2 is None:
        return {name1: self, name2: other}
    
    diff1 = []
    diff2 = []
    diff = {name1: diff1, name2: diff2}

    def _diff(obj1, obj2, parent=""):
        for k, v in obj1.items():
            if k not in obj2:
                _diff1 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                _diff2 = f"-{parent}.{k}" if parent else f"-{k}"
                diff1.append(_diff1)
                diff2.append(_diff2)
            elif v == obj2[k]:
                pass
                # diff[k] = None
            elif isinstance(v, (dict, DictConfig, Mapping)):
                _parent = f"{parent}.{k}" if parent else f"{k}"
                _diff(v, obj2[k], parent=_parent)
            else:
                _diff1 = f"≠{parent}.{k}: {v}" if parent else f"≠{k}: {v}"
                _diff2 = (
                    f"≠{parent}.{k}: {obj2[k]}" if parent else f"≠{k}: {obj2[k]}"
                )
                diff1.append(_diff1)
                diff2.append(_diff2)
        for k, v in obj2.items():
            if k not in obj1:
                _diff1 = f"-{parent}.{k}" if parent else f"-{k}"
                _diff2 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                diff1.append(_diff1)
                diff2.append(_diff2)

    _diff(obj1, obj2)
    return DictConfig(diff)


def without(obj, key: Union[str, list[str]]):
    """Return a copy of the namespace without the specified key."""
    _copy = deepcopy(obj)
    if isinstance(key, str):
        key = [key]
    [_copy.pop(k) for k in key]
    return _copy