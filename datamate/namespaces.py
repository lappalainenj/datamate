"""
This module exports `Namespace`, a `dict` that supports accessing items at
attributes, for convenience, and to better support static analysis.

It also exports`namespacify`, a function that recursively converts mappings and
Namespace-like containers in JSON-like objects to `Namespace`s.
"""

from typing import Mapping, Union, Dict, Any, List, get_origin
from copy import deepcopy
from numpy import ndarray
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig

import pandas as pd

__all__ = [
    "Namespace",
    "namespacify",
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

# -- Namespaces ----------------------------------------------------------------


class Namespace(Dict[str, Any]):
    """
    A `dict` that supports accessing items as attributes and comparison methods
    between multiple `Namespace`s.
    """

    def __init__(self, *args, **kwargs):
        base = dict(*args, **kwargs)
        converted = {k: namespacify(v) for k, v in base.items()}
        super().__init__(converted)

    def __dir__(self) -> List[str]:
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __getattr__(self, key: str) -> Any:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, val: object) -> None:
        dict.__setitem__(self, key, val)

    def __delattr__(self, key: str) -> None:
        dict.__delitem__(self, key)

    @property
    def __dict__(self) -> dict:  # type: ignore
        return self

    def __repr__(self) -> str:
        def single_line_repr(elem: object) -> str:
            if isinstance(elem, list):
                return "[" + ", ".join(map(single_line_repr, elem)) + "]"
            elif isinstance(elem, Namespace):
                return (
                    f"{elem.__class__.__name__}("
                    + ", ".join(f"{k}={single_line_repr(v)}" for k, v in elem.items())
                    + ")"
                )
            else:
                return repr(elem).replace("\n", " ")

        def repr_in_context(elem: object, curr_col: int, indent: int) -> str:
            sl_repr = single_line_repr(elem)
            if len(sl_repr) <= 80 - curr_col:
                return sl_repr
            elif isinstance(elem, list):
                return (
                    "[\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        repr_in_context(e, indent + 2, indent + 2) for e in elem
                    )
                    + "\n"
                    + " " * indent
                    + "]"
                )
            elif isinstance(elem, Namespace):
                return (
                    f"{elem.__class__.__name__}(\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        f"{k} = " + repr_in_context(v, indent + 5 + len(k), indent + 2)
                        for k, v in elem.items()
                    )
                    + "\n"
                    + " " * indent
                    + ")"
                )
            else:
                return repr(elem)

        return repr_in_context(self, 0, 0)

    def __eq__(self, other):
        return all_true(compare(namespacify(self), namespacify(other)))

    def __ne__(self, other):
        return not self.__eq__(other)


def namespacify(obj: object) -> Namespace:
    """
    Recursively convert mappings (item access only) and ad-hoc Namespaces
    (attribute access only) to `Namespace`s (both item and element access).
    """
    if isinstance(obj, (type(None), bool, int, float, str, type, bytes)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple, ListConfig)):
        return [namespacify(v) for v in obj]
    elif isinstance(obj, (ndarray)):
        return [namespacify(v.item()) for v in obj]
    elif isinstance(obj, Mapping):
        return Namespace({k: namespacify(obj[k]) for k in obj})
    elif get_origin(obj) is not None:
        return obj
    else:
        try:
            return namespacify(vars(obj))
        except TypeError as e:
            raise TypeError(f"namespacifying {obj} of type {type(obj)}: {e}.") from e
        

# -- Utility functions --------------------------------------------------------


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
    elif isinstance(obj, Namespace):
        return dict(
            (k, to_dict(v) if isinstance(v, dict) or isinstance(v, Namespace) else v)
            for k, v in obj.items()
        )
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
    if isinstance(cls, (dict, Namespace, DictConfig)):
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
        _obj1, _obj2 = deepcopy(obj1), deepcopy(obj2)
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
        return Namespace(out)
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
            elif isinstance(v, (dict, Namespace, DictConfig, Mapping)):
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
    return Namespace(diff)


def without(obj, key: Union[str, list[str]]):
    """Return a copy of the namespace without the specified key."""
    _copy = deepcopy(obj)
    if isinstance(key, str):
        key = [key]
    [_copy.pop(k) for k in key]
    return _copy