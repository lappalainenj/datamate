"""
This module exports `Namespace`, a `dict` that supports accessing items at
attributes, for convenience, and to better support static analysis.

It also exports`namespacify`, a function that recursively converts mappings and
Namespace-like containers in JSON-like objects to `Namespace`s.
"""

from typing import Any, Dict, List, Mapping
from copy import copy, deepcopy
from numpy import ndarray
from pathlib import Path

import pandas as pd

__all__ = ["Namespace", "namespacify"]

# -- Namespaces ----------------------------------------------------------------


class Namespace(Dict[str, Any]):
    """
    A `dict` that supports accessing items as attributes
    """

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

    def is_superset(self, superset):
        _keys_subset = set(self)
        _keys_superset = set(superset)
        return _keys_subset.issubset(_keys_superset)

    def is_value_matching_superset(self, subset):
        """
        Args:
            self (dict): potential superset
            subset (dict): potential subset

        Returns: True if subset is a subset of superset
            and all values match.
        """
        # namespacify for type coercion from array to list
        self = namespacify(self)
        subset = namespacify(subset)
        _keys_superset = set(self)
        _keys_subset = set(subset)
        _keys_are_superset = _keys_superset.issuperset(_keys_subset)
        _values_are_matching = all(
            self[key] == subset[key]
            for key in _keys_superset.intersection(_keys_subset)
        )

        return _keys_are_superset & _values_are_matching

    def to_df(self, name="", seperator="."):
        """Dict to flattened dataframe."""
        as_dict = self.to_dict()  # namespace need deepcopy method
        df = pd.json_normalize(as_dict, sep=seperator).T
        if name:
            df = df.rename({0: name}, axis=1)
        return df

    def diff(self, other, name1="self", name2="other"):
        """Diff two namespaces."""
        if self is None or other is None:
            return {name1: self, name2: other}
        # namespacify for type coercion from array to list
        _self = namespacify(self)
        other = namespacify(other)
        diff1 = []
        diff2 = []
        diff = {name1: diff1, name2: diff2}

        def _diff(self, other, parent=""):
            for k, v in self.items():
                if k not in other:
                    _diff1 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                    _diff2 = f"-{parent}.{k}" if parent else f"-{k}"
                    diff1.append(_diff1)
                    diff2.append(_diff2)
                elif v == other[k]:
                    pass
                    # diff[k] = None
                elif isinstance(v, Namespace):
                    _parent = f"{parent}.{k}" if parent else f"{k}"
                    _diff(v, other[k], parent=_parent)
                else:
                    _diff1 = f"≠{parent}.{k}: {v}" if parent else f"≠{k}: {v}"
                    _diff2 = (
                        f"≠{parent}.{k}: {other[k]}" if parent else f"≠{k}: {other[k]}"
                    )
                    diff1.append(_diff1)
                    diff2.append(_diff2)
            for k, v in other.items():
                if k not in self:
                    _diff1 = f"-{parent}.{k}" if parent else f"-{k}"
                    _diff2 = f"+{parent}.{k}: {v}" if parent else f"+{k}: {v}"
                    diff1.append(_diff1)
                    diff2.append(_diff2)

        _diff(_self, other)
        return namespacify(diff)

    def walk(self, _parents=tuple()):
        """Walks a dictionary, like os.walk(dir)."""
        _childs = tuple()
        _values = tuple()
        for key, value in self.items():
            parents = _parents + (key,)
            childs = (
                _childs + tuple(value.keys()) if isinstance(value, dict) else _childs
            )
            values = _values if isinstance(value, dict) else _values + (value,)
            yield parents, childs, values

            if isinstance(value, dict):
                yield from self.walk(value, _parents=parents)

    def equal_values(self, other):
        "Comparison of values, nested."
        # namespacify for type coercion from array to list
        _self = namespacify(self)
        other = namespacify(other)
        return compare(self, other)

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

    def to_dict(self):
        return to_dict(self)

    def depth(self):
        return depth(self)

    def pformat(self):
        return pformat(self)


def namespacify(obj: object) -> object:
    """
    Recursively convert mappings (item access only) and ad-hoc Namespaces
    (attribute access only) to `Namespace`s (both item and element access).
    """
    if isinstance(obj, (type(None), bool, int, float, str, type, bytes)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [namespacify(v) for v in obj]
    elif isinstance(obj, (ndarray)):
        return [namespacify(v.item()) for v in obj]
    elif isinstance(obj, Mapping):
        return Namespace({k: namespacify(obj[k]) for k in obj})
    else:
        try:
            return namespacify(vars(obj))
        except TypeError as e:
            raise TypeError(f"namespacifying {obj} of type {type(obj)}: {e}.") from e


def to_dict(cls):
    """
    Recursively converts a Namespace to a dictionary.
    """
    if isinstance(cls, Namespace):
        return {k: to_dict(v) for k, v in cls.__dict__.items()}
    elif isinstance(cls, (list, tuple)):
        return type(cls)(to_dict(v) for v in cls)
    else:
        return cls


def depth(cls):
    if isinstance(cls, (dict, Namespace)):
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
        return Namespace(out)
    elif type(obj1) != type(obj2):
        return False
