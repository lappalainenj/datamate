"""
This module exports the `Directory` class, an array- and metadata-friendly view
into a directory.

Instances of the base Directory class have methods to simplify reading/writing
collections of arrays.

This module also exports `ArrayFile` a descriptor protocol intended to be used
as attribute type annotations within `Directory` subclass definition.
"""
import os
import warnings
import itertools
import json
from pathlib import Path
import shutil
import functools
import threading
from time import sleep
import inspect
from numbers import Number
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Dict,
    Union,
    cast,
    get_origin,
)
from typing_extensions import Protocol
import datetime
from traceback import format_tb


from contextlib import contextmanager

import h5py as h5
import numpy as np
from pandas import DataFrame

from datamate.namespaces import (
    Namespace,
    namespacify,
    is_disjoint,
    is_superset,
    to_dict,
)

__all__ = ["Directory", "ArrayFile"]

# -- Custom Errors and Warnings ------------------------------------------------


class ConfigWarning(Warning):
    pass


class ModifiedWarning(Warning):
    pass


class ModifiedError(Exception):
    pass


class ImplementationWarning(Warning):
    pass


class ImplementationError(Exception):
    pass


# -- Static type definitions ---------------------------------------------------


class ArrayFile(Protocol):
    """
    A property that corresponds to a single-array HDF5 file
    """

    def __get__(self, obj: object, type_: Optional[type]) -> h5.Dataset:
        ...

    def __set__(self, obj: object, val: object) -> None:
        ...


NoneType = type(None)


# -- Root Directory directory management ----------------------------------------

context = threading.local()
context.enforce_config_match = True
context.check_size_on_init = False
context.verbosity_level = 2
# context.in_memory = False


def set_root_dir(root_dir: Optional[Path]) -> None:
    """
    Set the directory in which to search for Directorys.
    """
    context.root_dir = Path(root_dir) if root_dir is not None else Path(".")


def get_root_dir() -> Path:
    """
    Return the current Directory search directory.
    """
    return getattr(context, "root_dir", Path("."))


def root(root_dir: Union[str, Path, NoneType] = None):
    """Decorates a callable to fix its individual root directory.

    root_dir: optional root directory that will be set at execution of the
        callable. Optional. Default is None which corresponds to get_root_dir().

    Example:
        @root("/path/to/this/individual/dir")
        class MyDirectory(Directory):
            ...

        dir = MyDirectory(...)
        dir.path.parent == "/path/to/this/individual/dir"

    Note, when decorating with `root`, the decorator has
    precedence over changing the root dir with `set_root_dir` in the outer scope.
    However, to still change the root directory for a decorated callable from
    an outer scope, use `set_root_context` instead.

    Example:
        with set_root_context(new_dir):
            dir = MyDirectory(...)
    """

    def decorator(callable):
        if inspect.isfunction(callable):

            @functools.wraps(callable)
            def function(*args, **kwargs):
                _root_dir = get_root_dir()
                within_context = getattr(context, "within_root_context", False)
                # case 1: root_dir provided
                if root_dir is not None and not within_context:
                    set_root_dir(root_dir)
                # case 2: root_dir provided and not within context
                # case 3: root_dir provided and within context
                # case 4: root dir not provided and not within context
                # case 5: root dir not provided and within context
                else:
                    set_root_dir(_root_dir)
                _return = callable(*args, **kwargs)
                set_root_dir(_root_dir)
                return _return

            return function
        elif inspect.isclass(callable):
            new = callable.__new__

            @functools.wraps(callable)
            def function(*args, **kwargs):
                _root_dir = get_root_dir()
                within_context = getattr(context, "within_root_context", False)
                # case 1: root_dir provided
                if root_dir is not None and not within_context:
                    set_root_dir(root_dir)
                # case 2: root_dir provided and not within context
                # case 3: root_dir provided and within context
                # case 4: root dir not provided and not within context
                # case 5: root dir not provided and within context
                else:
                    set_root_dir(_root_dir)
                _return = new(*args, **kwargs)
                set_root_dir(_root_dir)
                return _return

            callable.__new__ = function

            return callable
        else:
            raise ValueError

    return decorator


@contextmanager
def set_root_context(root_dir: Union[str, Path, NoneType] = None):
    """Set root directory within a context and revert after.

    Example:
        with set_root_context(dir):
            Directory(config)

    Note, takes precendecence over all other methods to control the root
        directory.
    """
    _root_dir = get_root_dir()
    set_root_dir(root_dir)
    context.within_root_context = True
    try:
        yield
    finally:
        set_root_dir(_root_dir)
        context.within_root_context = False


# @contextmanager
# def in_memory():
#     """Set in_memory mode within a context and revert after to debug a Directory.
#     """
#     _in_memory = getattr(context, "in_memory", False)
#     context.in_memory = True
#     try:
#         yield
#     finally:
#         context.in_memory = _in_memory


def enforce_config_match(enforce: bool) -> None:
    """
    Enforce error if configs are not matching.

    Defaults to True.

    Configs are compared, when initializing a directory
    from an existing path and configuration.
    """
    context.enforce_config_match = enforce


def check_size_on_init(enforce: bool) -> None:
    """
    Switch size warning on/off.

    Defaults to False.

    Note: checking the size of a directory is slow, therefore this should
    be used only consciously.
    """
    context.check_size_on_init = enforce


def get_check_size_on_init() -> bool:
    return context.check_size_on_init


def set_verbosity_level(level: int) -> None:
    """
    Set verbosity level of representation for Directorys.

    0: only the top level directory name and the last modified date are shown.
    1: maximally 2 levels and 25 lines are represented.
    2: all directorys and files are represented.

    Defaults to 2.
    """
    context.verbosity_level = level


def set_scope(scope: Optional[Dict[str, type]]) -> None:
    """
    Set the scope used for "type" field resolution.
    """
    if hasattr(context, "scope"):
        del context.scope
    if scope is not None:
        context.scope = scope


def get_scope() -> Dict[str, type]:
    """
    Return the scope used for "type" field resolution.
    """
    return context.scope if hasattr(context, "scope") else get_default_scope()


def get_default_scope(cls: object = None) -> Dict[str, type]:
    """
    Return the default scope used for "type" field resolution.
    """

    def subclasses(t: type) -> Iterator[type]:
        yield from itertools.chain([t], *map(subclasses, t.__subclasses__()))

    cls = cls or Directory
    scope: Dict[str, type] = {}
    for t in subclasses(cls):
        scope[t.__qualname__] = t
    return scope


def reset_scope(cls: object = None) -> None:
    """
    Reset the scope to the default scope.
    """
    set_scope(get_default_scope(cls))


# -- Directorys -----------------------------------------------------------------


class NonExistingDirectory(type):
    """Directory metaclass to allow create non-existing Directory instances."""

    def __call__(cls, *args, **kwargs):
        return cls.__new__(cls, *args, **kwargs)


class Directory(metaclass=NonExistingDirectory):
    """
    An array- and metadata-friendly view into a directory

    Arguments:
        path Union[str, Path]: The path at which the Directory is, or should be,
            stored, can be relative to the current `root_dir`.
        config Dict[str, object]: The configuration of the Directory.
            When including a "type" field indicates the type of Directory to
            search for and construct in the scope. Note, the config can be
            unpacked into the constructor as keyword arguments.

    Valid constructors:
            To auto-name Directorys relative to `root_dir`:
            - Directory()
            - Directory(config: Dict[str, object])

            To name Directorys relative to `root_dir` or absolute:
            - Directory(path: Union[str, Path])
            - Directory(path: Union[str, Path], config: Dict[str, object])
            Note, config can also be passed as keyword arguments after
            `path`.

    If __init__ is implemented, it will be called to build the Directory.

    If only `path` is provided, the corresponding Directory is
    returned. It will be empty if `path` points to an empty or nonexistent
    directory.

    If only `config` is provided, it will search the current `root_dir`
    for a matching directory, and return a Directory pointing there if it
    exists. Otherwise, a new Directory will be constructed at the top level of
    the `root_dir`.

    If both `path` and `config` are provided, it will return the Directory
    at `path`, building it if necessary. If `path` points to an existing
    directory that is not a sucessfully built Directory matching `config`, an
    error is raised.

    Attributes:
        path (Path): The path at which the Directory is, or should be, stored.
        config (Dict[str, object]): The configuration of the Directory.

    After instantiation, Directorys act as string-keyed mutable dictionaries,
    containing three types of entries: `ArrayFile`s, `Path`s, and other `Directory`s.

    `ArrayFile`s are single-entry HDF5 files, in SWMR mode. Array-like numeric
    and byte-string data (valid operands of `numpy.asarray`) written into an
    Directory via `__setitem__`, `__setattr__`, or `extend` is stored as an array
    file.

    `Path`s are non-array files, presumed to be encoded in a format that
    is not understood. They are written and read as normal
    `Path`s, which support simple text and byte I/O, and can be passed to more
    specialized libraries for further processing.

    `Directory` entries are returned as properly subtyped Directorys, and can be
    created, via `__setitem__`, `__setattr__`, or `extend`, from existing
    Directorys or (possibly nested) dictionaries of arrays.
    """

    class Config(Protocol):
        """
        `Config` classes are intended to be interface definitions. They are
        used to define the structure of the `config` argument to the
        `Directory` constructor, and to provide type hints for the `config`
        attribute of `Directory` instances.
        """

        pass

    path: Path
    config: Config

    def __new__(_type, *args: object, **kwargs: object) -> Any:
        path, config = _parse_directory_args(args, kwargs)
        cls = _directory(_type)
        _check_implementation(cls)

        defaults = get_defaults(cls)

        if config is None and defaults:  # and _implements_init(cls):
            # to initialize from defaults if no config or path is provided
            if path is None:
                config = defaults
            # to initialize from defaults if no config and empty path is provided
            elif path is not None and not path.exists():
                config = defaults
            # if a non-empty path is provided, we cannot initialize from defaults
            else:
                pass
        # breakpoint()
        if path is not None and config is None:
            cls = _directory_from_path(cls, _resolve_path(path))
        elif path is None and config is not None:
            cls = _directory_from_config(cls, config)
        elif path is not None and config is not None:
            cls = _directory_from_path_and_config(cls, _resolve_path(path), config)
        elif path is None and config is None:
            if _implements_init(cls):
                # raise ValueError("no configuration provided")
                pass

        if context.check_size_on_init:
            cls.check_size()

        return cls

    def __init__(self):
        """Implement to compile `Directory` from a configuration.

        Note, subclasses can either implement Config to determine the interface,
        types and defaults of `config`, or implement `__init__` with keyword
        arguments to determine the interface, types and defaults of `config`.
        In case both are implemented, the config is created from the joined
        interface of both as long as defaults are not conflicting.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__doc__ = _auto_doc(cls)

    @property
    def meta(self) -> Namespace:
        """
        The metadata stored in `{self.path}/_meta.yaml`
        """
        return read_meta(self.path)

    @property
    def config(self):
        return self.meta.config or self._config

    @property
    def status(self):
        return self.meta.status

    # -- MutableMapping methods ----------------------------

    def __len__(self) -> int:
        """
        Returns the number of public files in `self.path`

        Non-public files (files whose names start with "_") are not counted.
        """
        return sum(1 for _ in self.path.glob("[!_]*"))

    def __iter__(self) -> Iterator[str]:
        """
        Yields field names corresponding to the public files in `self.path`

        Entries it understands (subdirectories and HDF5 files) are yielded
        without extensions. Non-public files (files whose names start with "_")
        are ignored.
        """
        for p in self.path.glob("[!_]*"):
            yield p.name[:-3] if p.suffix == ".h5" else p.name

    def __copy__(self):
        return Directory(self.path)

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def keys(self) -> Iterator[str]:
        return self.__iter__()

    def items(self) -> Iterator[Tuple[str, ArrayFile]]:
        for key in self.keys():
            yield (key, self[key])

    @classmethod
    def from_df(cls, df: DataFrame, dtypes: dict, *args, **kwargs):
        directory = Directory.__new__(Directory, *args, **kwargs)
        directory.update(
            {column: df[column].values.astype(dtypes[column]) for column in df.columns}
        )
        return directory

    def update(self, other, suffix: str = "") -> None:
        """
        Updates self with items of other and appends an optional suffix.
        """
        for key in other:
            if key + suffix not in self:
                self[key + suffix] = other[key]

    def move(self, dst):
        """Move directory to dst."""
        shutil.move(self.path, dst)
        return Directory(dst)

    def rmtree(self, y_n=None):
        reply = y_n or input(f"delete {self.path} recursively, y/n?")
        if reply.lower() == "y":
            shutil.rmtree(self.path, ignore_errors=True)

    def _rebuild(self, y_n=None):
        self.rmtree(y_n)
        _build(self)

    def __truediv__(self, other):
        return self.__getitem__(other)

    def __getitem__(self, key: str) -> Any:
        """
        Returns an `ArrayFile`, `Path`, or `Directory` corresponding to
        `self.path/key`

        HDF5 files are returned as `ArrayFile`s, other files are returned as
        `Path`s, and directories and nonexistent entries are returned as
        (possibly empty) `Directory`s.

        Attribute access syntax is also supported, and occurrences of "__" in
        `key` are transformed into ".", to support accessing encoded files as
        attributes (i.e. `Directory['name.ext']` is equivalent to
        `Directory.name__ext`).
        """
        # if context.in_memory:
        #     return object.__getattribute__(self, key)

        try:
            # to catch cases where key is an index to a reference to an h5 file.
            # this will yield a TypeError because Path / slice does not work.
            path = self.path / key
        except TypeError as e:
            if not self.path.exists():
                raise (
                    AssertionError(
                        f"Indexing {self.path.name} at {key} not possible for"
                        f" Directory at {self.path.parent}. File "
                        f"{self.path.name}.h5 does not exist."
                    )
                )
            raise e

        # Return an array.
        if path.with_suffix(".h5").is_file():
            return _read_h5(path.with_suffix(".h5"))

        # Return the path to a file.
        elif path.is_file():
            return path

        # Return a subrecord
        else:
            return Directory(path)

    def __setitem__(self, key: str, val: object) -> None:
        """
        Writes an `ArrayFile`, `Path`, or `Directory` to `self.path/key`

        `np.ndarray`-like objects are written as `ArrayFiles`, `Path`-like
        objects are written as `Path`s, and string-keyed mappings are
        written as subDirectorys.

        Attribute access syntax is also supported, and occurrences of "__" in
        `key` are transformed into ".", to support accessing encoded files as
        attributes (i.e. `Directory['name.ext'] = val` is equivalent to
        `Directory.name__ext = val`).
        """
        # if context.in_memory:
        #     object.__setattr__(self, key, val)
        #     return

        path = self.path / key

        # Copy an existing file or directory.
        if isinstance(val, Path):
            if os.path.isfile(val):
                _copy_file(path, val)
            elif os.path.isdir(val):
                _copy_dir(path, val)

        # Write a subDirectory.
        elif isinstance(val, (Mapping, Directory)):
            assert path.suffix == ""
            MutableMapping.update(Directory(path), val)  # type: ignore

        # Write an array.
        else:
            assert path.suffix == ""
            try:
                _write_h5(path.with_suffix(".h5"), val)
            except TypeError as err:
                raise TypeError(
                    format_tb(err.__traceback__)[0]
                    + err.args[0]
                    + f"\nYou're trying to store {val} which cannot be converted to h5-file in {path}."
                    + "\nFor reference of supported types, see https://docs.h5py.org/en/stable/faq.html?highlight=types#numpy-object-types"
                    + "\nE.g. NumPy unicode strings must be converted to 'S' strings and back:"
                    + "\nfoo.bar = array.astype('S') to store and foo.bar[:].astype('U') to retrieve."
                ) from None

        if self.config is not None and self.status == "done":
            # Track if a Directory has been modified past __init__
            self._modified_past_init(True)

    def __delitem__(self, key: str) -> None:
        """
        Deletes the entry at `self.path/key`

        Attribute access syntax is also supported, and occurrences of "__" in
        `key` are transformed into ".", to support accessing encoded files as
        attributes (i.e. `del Directory['name.ext']` is equivalent to
        `del Directory.name__ext`).
        """
        # if context.in_memory:
        #     object.__delitem__(self, key)
        #     return
        path = self.path / key

        # Delete an array file.
        if path.with_suffix(".h5").is_file():
            path.with_suffix(".h5").unlink()

        # Delete a non-array file.
        elif path.is_file():
            path.unlink()

        # Delete a Directory.
        else:
            shutil.rmtree(path, ignore_errors=True)

    def extend(self, key: str, val: object) -> None:
        """
        Extends an `ArrayFile`, `Path`, or `Directory` at `self.path/key`

        Extending `ArrayFile`s performs concatenation along the first axis,
        extending `Path`s performs byte-level concatenation, and
        extending subDirectorys extends their fields.

        Files corresponding to `self[key]` are created if they do not already
        exist.
        """
        # if context.in_memory:
        #     self.__setitem__(key, np.append(self.__getitem__(key), val, axis=0))

        path = self.path / key

        # Append an existing file.
        if isinstance(val, Path):
            assert path.suffix != ""
            _extend_file(path, val)

        # Append a subDirectory.
        elif isinstance(val, (Mapping, Directory)):
            assert path.suffix == ""
            for k in val:
                Directory(path).extend(k, val[k])

        # Append an array.
        else:
            assert path.suffix == ""
            _extend_h5(path.with_suffix(".h5"), val)

        if self.config is not None and self.status == "done":
            # Track if a Directory has been modified past __init__
            self._modified_past_init(True)

    # --- Views ---

    def __repr__(self):
        if context.verbosity_level == 1:
            string = tree(
                self.path,
                last_modified=True,
                level=2,
                length_limit=25,
                verbose=True,
                not_exists_msg="empty",
            )
        elif context.verbosity_level == 0:
            string = tree(
                self.path,
                last_modified=True,
                level=1,
                length_limit=0,
                verbose=False,
                not_exists_msg="empty",
            )
        else:
            string = tree(
                self.path,
                level=-1,
                length_limit=None,
                last_modified=True,
                verbose=True,
                limit_to_directories=False,
            )
        return string

    def tree(
        self,
        level=-1,
        length_limit=None,
        verbose=True,
        last_modified=True,
        limit_to_directories=False,
    ):
        print(
            tree(
                self.path,
                level=level,
                length_limit=length_limit,
                last_modified=last_modified,
                verbose=verbose,
                limit_to_directories=limit_to_directories,
            )
        )

    # -- Attribute-style element access --------------------

    def __getattr__(self, key: str) -> Any:
        return self.__getitem__(key.replace("__", "."))

    def __setattr__(self, key: str, value: object) -> None:
        # Fix autoreload related effect.
        if key.startswith("__") and key.endswith("__"):
            object.__setattr__(self, key, value)
            return
        self.__setitem__(key.replace("__", "."), value)

    def __delattr__(self, key: str) -> None:
        self.__delitem__(key.replace("__", "."))

    # -- Attribute preemption, for REPL autocompletion -----

    def __getattribute__(self, key: str) -> Any:
        if key in object.__getattribute__(self, "_cached_keys"):
            try:
                object.__setattr__(self, key, self[key])
            except KeyError:
                object.__delattr__(self, key)
                object.__getattribute__(self, "_cached_keys").remove(key)
        return object.__getattribute__(self, key)

    def __dir__(self) -> List[str]:
        for key in self._cached_keys:
            object.__delattr__(self, key)
        self._cached_keys.clear()

        for key in set(self).difference(object.__dir__(self)):
            object.__setattr__(self, key, self[key])
            self._cached_keys.add(key)

        return cast(list, object.__dir__(self))

    # -- Convenience methods

    def _override_config(self, config, status=None):
        """Overriding config stored in _meta.yaml.

        config (Dict): update for meta.config
        status (str): status if config did not exist before, i.e. _overrid_config
            is used to store a _meta.yaml for the first time instead of build.
        """
        meta_path = self.path / "_meta.yaml"

        def write_meta(**kwargs):
            meta_path.write_text(json.dumps(_identify_elements(kwargs)))

        current_config = self.config
        if current_config is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        f"Overriding config. Diff is:"
                        f'{config.diff(current_config, name1="passed", name2="stored")}'
                    ),
                    ConfigWarning,
                    stacklevel=2,
                )
            write_meta(config=config, status="overridden")
        else:
            write_meta(config=config, status=status or self.status)

    def _override_status(self, status):
        meta_path = self.path / "_meta.yaml"

        def write_meta(**kwargs):
            meta_path.write_text(json.dumps(_identify_elements(kwargs)))

        current_status = self.status
        if current_status is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (f"Overriding status {current_status} to {status}"),
                    ConfigWarning,
                    stacklevel=2,
                )
        write_meta(config=self.config, status=status)

    def _modified_past_init(self, is_modified):
        meta_path = self.path / "_meta.yaml"

        def write_meta(**kwargs):
            meta_path.write_text(json.dumps(_identify_elements(kwargs)))

        if is_modified:
            write_meta(config=self.config, status=self.status, modified=True)

    def check_size(self, warning_at=20 * 1024**3, print_size=False) -> None:
        """Prints the size of the directory in bytes."""
        return check_size(self.path, warning_at, print_size)

    def to_df(self, dtypes: dict = None) -> DataFrame:
        """
        Returns a DataFrame from all equal length, single-dim .h5 datasets in self.path.
        """
        # to cache the dataframe that is expensive to create.
        try:
            return object.__getattribute__(self, "_as_df")
        except:
            object.__setattr__(self, "_as_df", directory_to_df(self, dtypes))
            return self.to_df()

    def to_dict(self) -> DataFrame:
        """
        Returns a DataFrame from all equal length, single-dim .h5 datasets in self.path.
        """
        # to cache the dict that is expensive to create.
        try:
            return object.__getattribute__(self, "_as_dict")
        except:
            object.__setattr__(self, "_as_dict", directory_to_dict(self))
            return self.to_dict()

    def mtime(self):
        return datetime.datetime.fromtimestamp(self.path.stat().st_mtime)

    @property
    def parent(self):
        return Directory(self.path.absolute().parent)

    def _count(self) -> int:
        root = self.path
        count = 0
        for i in itertools.count():
            dst = root / f"{i:04x}"
            if dst.exists():
                count += 1
            else:
                return count
        return count

    def _next(self) -> int:
        root = self.path
        dst = root / f"{self._count():04x}"
        assert not dst.exists()
        return Directory(dst, self.config)

    def _clear_filetype(self, suffix: str) -> None:
        """
        Delete files ending with suffix in the current directory path
        """
        for file in self.path.iterdir():
            if file.is_file() and file.suffix == suffix:
                file.unlink()


# -- Directory construction -----------------------------------------------------


def merge(dict1, dict2):
    merged = {}
    if is_disjoint(dict1, dict2):
        merged.update(dict1)
        merged.update(dict2)
    elif is_superset(dict1, dict2):
        merged.update(dict1)
    elif is_superset(dict2, dict1):
        merged.update(dict2)
    else:
        raise ValueError(f"merge conflict: {dict1} and {dict2}")
    return merged


def get_defaults(cls: Directory):
    try:
        return merge(get_defaults_from_Config(cls), get_defaults_from_init(cls))
    except ValueError as e:
        raise ValueError("conflicting defaults") from e


def get_defaults_from_Config(cls: Union[type, Directory]):
    cls = cls if isinstance(cls, type) else type(cls)
    if "Config" in cls.__dict__:
        return {
            k: v
            for k, v in cls.Config.__dict__.items()
            if not (k.startswith("_") or (k.startswith("__") and k.endswith("__")))
        }
    return {}


def get_defaults_from_init(cls: Directory):
    cls = cls if isinstance(cls, type) else type(cls)
    signature = inspect.signature(cls.__init__)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default != inspect._empty
    }
    return defaults


def get_annotations(cls: Union[type, Directory]):
    return merge(get_annotations_from_Config(cls), get_annotations_from_init(cls))


def get_annotations_from_Config(cls: Union[type, Directory]):
    cls = cls if isinstance(cls, type) else type(cls)
    if "Config" in cls.__dict__:
        annotations = getattr(cls.Config, "__annotations__", {})
        return annotations
    return {}


def get_annotations_from_init(cls: Directory):
    cls = cls if isinstance(cls, type) else type(cls)
    return {k: v for k, v in cls.__init__.__annotations__.items() if v is not None}


def _parse_directory_args(
    args: Tuple[object, ...], kwargs: Mapping[str, object]
) -> Tuple[Optional[Path], Optional[Mapping[str, object]]]:
    """
    Return `(path, conf)` or raise an error.
    """
    # ()
    if len(args) == 0 and len(kwargs) == 0:
        return None, None

    # (conf)
    elif len(args) == 1 and isinstance(args[0], Mapping) and len(kwargs) == 0:
        return None, dict(args[0])

    # (**conf)
    elif len(args) == 0 and len(kwargs) > 0:
        return None, kwargs

    # (path)
    elif len(args) == 1 and isinstance(args[0], Path) and len(kwargs) == 0:
        return Path(args[0]), None

    # (str)
    elif len(args) == 1 and isinstance(args[0], str) and len(kwargs) == 0:
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), None
        root_dir = get_root_dir()
        return root_dir / args[0], None

    # (path, conf)
    elif (
        len(args) == 2
        and isinstance(args[0], Path)
        and isinstance(args[1], Mapping)
        and len(kwargs) == 0
    ):
        return Path(args[0]), dict(args[1])

    # (str, conf)
    elif (
        len(args) == 2
        and isinstance(args[0], str)
        and isinstance(args[1], Mapping)
        and len(kwargs) == 0
    ):
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), dict(args[1])
        root_dir = get_root_dir()
        return root_dir / args[0], dict(args[1])

    # (path, **conf)
    elif len(args) == 1 and isinstance(args[0], Path) and len(kwargs) > 0:
        return Path(args[0]), kwargs

    # (str, **conf)
    elif len(args) == 1 and isinstance(args[0], str) and len(kwargs) > 0:
        if args[0][0] in [".", "..", "~", "@"]:
            return Path(args[0]), kwargs
        root_dir = get_root_dir()
        return root_dir / args[0], kwargs

    # <invalid signature>
    else:
        raise TypeError(
            "Invalid argument types for the `Directory` constructor.\n"
            "Valid signatures:\n"
            "\n"
            "    - Directory()\n"
            "    - Directory(config: Dict[str, object])\n"
            "    - Directory(path: Path)\n"
            "    - Directory(path: Path, config: Dict[str, object])\n"
            "    - Directory(name: str)\n"
            "    - Directory(name: str, config: Dict[str, object])"
            "Note, config can also be passed as keyword arguments after "
            "`path` or `name`. `name` is relative to the root directory."
        )


def _implements_init(cls: Directory) -> bool:
    """True if the class implements `__init__`."""
    return inspect.getsource(cls.__init__).split("\n")[-2].replace(" ", "") != "pass"


def _check_implementation(cls: Directory):
    defaults = get_defaults(cls)
    # check if Config only has annotations, no defaults
    annotations = get_annotations(cls)

    if _implements_init(cls) and not defaults and not annotations:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                (
                    f"The Directory {type(cls)} implements __init__ to write data"
                    " but specifies no configuration."
                ),
                ImplementationWarning,
                stacklevel=2,
            )


def _directory(cls: type) -> Directory:
    """
    Return a new Directory at the root of the file tree.
    """
    directory = _forward_subclass(cls, {})
    path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _directory_from_path(cls: Directory, path: Path) -> Directory:
    """
    Return a Directory corresponding to the file tree at `path`.

    An error is raised if the type recorded in `_meta.yaml`, if any, is not a
    subtype of `cls`.
    """

    config = read_meta(path).config or {}
    written_type = get_scope().get(config.get("type", None), None)

    if path.is_file():
        raise FileExistsError(f"{path} is a file.")

    # if context.enforce_config_match:

    if not path.is_dir():
        if _implements_init(cls) and not get_defaults(cls):
            raise FileNotFoundError(
                f"cannot initialize {path}. It does not yet exist"
                f" and no config was provided to initialize it."
            )
    else:
        pass

    if written_type is not None and not issubclass(written_type, type(cls)):
        raise FileExistsError(
            f"{path} is a {written_type.__module__}.{written_type.__qualname__}"
            f", not a {cls.__module__}.{cls.__qualname__}."
        )

    # if context.enforce_config_match:
    directory = _forward_subclass(type(cls), config)
    # else:
    #     directory = _forward_subclass(type(cls), {})

    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _directory_from_config(cls: Directory, conf: Mapping[str, object]) -> Directory:
    """
    Find or build a Directory with the given type and config.
    """
    directory = _forward_subclass(type(cls), conf)
    new_dir_path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    config = Namespace(**directory._config)

    def _new_directory():
        object.__setattr__(directory, "path", new_dir_path)
        # don't build cause only the type field is populated
        if list(config.keys()) == ["type"]:
            return directory
        # don't build cause the config matches the defaults and init is not implemented
        if not _implements_init(cls) and config.without("type") == get_defaults(cls):
            return directory
        # catches FileExistsError for the case when two processes try to
        # build the same directory simultaneously
        try:
            _build(directory)
        except FileExistsError:
            return _directory_from_config(cls, conf)
        return directory

    for path in Path(get_root_dir()).glob("*"):
        meta = read_meta(path)

        if meta.config == config:
            if getattr(meta, "modified", False):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        (
                            f"The Directory {path} has been modified after being build."
                            + f"\nBuilding a new directory {new_dir_path} to prevent that the files you would get from the object"
                            + " mismatch the files you would expect based on your __init__ and configuration."
                            + "\nYou can circumvent this"
                            + " by e.g. using the explicit path as constructor (see Directory docs)."
                        ),
                        ModifiedWarning,
                        stacklevel=2,
                    )
                return _new_directory()
            # raise ModifiedError(
            #             f"The Directory {path} has been modified after being build."
            #             + "\nThis could mean that the files you would get from the object"
            #             + " mismatch the files you would expect based on your configuration."
            #             + "\nYou can circumvent this error"
            #             + " by e.g. using the path as constructor"
            #             + " as explained in the Directory docs."
            #         )

            while meta.status == "running":
                sleep(0.01)
                meta = read_meta(path)
            if meta.status == "done":
                object.__setattr__(directory, "path", path)
                return directory
    return _new_directory()


def _directory_from_path_and_config(
    cls: Directory, path: Path, conf: Mapping[str, object]
) -> Directory:
    """
    Find or build a Directory with the given type, path, and config.
    """
    directory = _forward_subclass(type(cls), conf)
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)

    if path.exists():
        meta = read_meta(path)
        config = Namespace({"type": _identify(type(directory)), **directory._config})
        if meta.config != config:
            with warnings.catch_warnings():
                if context.enforce_config_match:
                    raise FileExistsError(
                        f'"{directory.path}" (incompatible config):\n'
                        f'{config.diff(meta.config, name1="passed", name2="stored")}'
                    )
                else:
                    warnings.simplefilter("always")
                    warnings.warn(
                        (
                            f'"{directory.path}" (incompatible config):\n'
                            f'{config.diff(meta.config, name1="passed", name2="stored")}'
                        ),
                        Warning,
                        stacklevel=2,
                    )
        while meta.status == "running":
            sleep(0.01)
            meta = read_meta(path)
        if directory.meta.status == "stopped":
            raise FileExistsError(f'"{directory.path}" was stopped mid-build.')
    else:
        # don't build cause only the type field is populated
        if list(directory._config.keys()) == ["type"]:
            return directory
        # don't build cause the config matches the defaults and init is not implemented
        if not _implements_init(cls) and directory._config.without(
            "type"
        ) == get_defaults(cls):
            return directory
        # catches FileExistsError for the case when two processes try to
        # build the same directory simultaneously
        try:
            _build(directory)
        except FileExistsError:
            return _directory_from_path_and_config(cls, path, conf)
    return directory


def _build(directory: Directory) -> None:
    """
    Create parent directories, invoke `Directory.__init__`, and store metadata.
    """

    if directory.path.exists() and directory.status == "done":
        return
    elif directory.path.exists() and directory.status == "running":
        sleep(0.01)
        return _build(directory)

    directory.path.mkdir(parents=True)

    meta_path = directory.path / "_meta.yaml"
    config = Namespace(**directory._config)
    write_meta = lambda **kwargs: meta_path.write_text(
        json.dumps(_identify_elements(kwargs))
    )

    write_meta(config=config, status="running")

    try:
        if callable(getattr(type(directory), "__init__", None)):
            n_build_args = directory.__init__.__code__.co_argcount
            # case 1: __init__(self)
            if n_build_args <= 1:
                build_args = []
                build_kwargs = {}
            # case 2: __init__(self, config)
            elif n_build_args == 2:
                build_args = [directory._config]
                build_kwargs = {}
            # case 3: __init__(self, foo=1, bar=2) to specify defaults and config
            else:
                kwargs = namespacify(get_defaults_from_init(directory))
                assert kwargs
                build_args = []
                build_kwargs = {k: directory._config[k] for k in kwargs}
            directory.__init__(*build_args, **build_kwargs)

        write_meta(config=config, status="done")
    except BaseException as e:
        write_meta(config=config, status="stopped")
        raise e


def call_signature(cls):
    signature = """

Example call signature:
    {}
    {}
Note, these use the `Directory(config: Dict[str, object])` constructor and are
inferred from defaults and annotations, i.e. they are equivalent to constructing
without arguments."""
    defaults = to_dict(get_defaults(cls))
    string = ""
    for key, val in defaults.items():
        string += f"{key}={val}, "
    if string.endswith(", "):
        string = string[:-2]
    # variant 1: unpacking config kwargs
    signature1 = ""
    if string:
        signature1 = f"{cls.__qualname__}({string})"
    # variant 2: whole config
    signature2 = ""
    if defaults:
        signature2 = f"{cls.__qualname__}({defaults})"

    if signature1 and signature2:
        return signature.format(signature1, signature2)

    return signature.format("(specify defaults for auto-doc of call signature)", "")


def type_signature(cls):
    signature = """

    Types of config elements:
       {}

    """
    annotations = to_dict(get_annotations(cls))

    def qualname(annotation):
        origin = get_origin(annotation)
        if origin:
            return repr(annotation)
        return annotation.__qualname__

    signature1 = ""
    for key, val in annotations.items():
        signature1 += f"{key}: {qualname(val)}, "
    if signature1.endswith(", "):
        signature1 = signature1[:-2]
    if signature1:
        return signature.format(signature1)
    return signature.format("(annotate types for auto-doc of type signature)")


def _auto_doc(cls: type, cls_doc=True, base_doc=False):
    docstring = "{}{}{}{}"
    if isinstance(cls, Directory):
        cls = type(cls)
    call_sig = call_signature(cls)
    type_sig = type_signature(cls)

    _cls_doc = ""
    if cls_doc and cls.__doc__:
        _cls_doc = cls.__doc__

    _base_doc = ""
    if base_doc and cls.__base__.__doc:
        _base_doc = cls.__base__.__doc__

    return docstring.format(_cls_doc, call_sig, type_sig, _base_doc)


def _resolve_path(path: Path) -> Path:
    """
    Dereference ".", "..", "~", and "@".
    """
    if path.parts[0] == "@":
        path = get_root_dir() / "/".join(path.parts[1:])
    return path.expanduser().resolve()


def _new_directory_path(type_: type) -> Path:
    """
    Generate an unused path in the Directory root directory.
    """
    root = Path(get_root_dir())
    type_name = _identify(type_)
    for i in itertools.count():
        dst = root / f"{type_name}_{i:04x}"
        if not dst.exists():
            return dst.absolute()
    assert False  # for MyPy


def check_size(path: Path, warning_at=20 * 1024**3, print_size=False) -> None:
    """Prints the size of the directory at path and warns if it exceeds warning_at."""

    def sizeof_fmt(num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, "Yi", suffix)

    def get_size(start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except FileNotFoundError:
                        pass
        return total_size

    size_in_bytes = get_size(path)
    if print_size:
        print(f"{sizeof_fmt(size_in_bytes)}")
    if size_in_bytes >= warning_at:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                f"This directory {path.name} occupies {sizeof_fmt(size_in_bytes)} of disk space.",
                ResourceWarning,
                stacklevel=2,
            )
    return size_in_bytes


def _forward_subclass(cls: type, config: object = {}) -> object:
    # Coerce `config` to a `dict`.
    config = dict(
        config if isinstance(config, Mapping) else getattr(config, "__dict__", {})
    )

    # Perform subclass forwarding.
    cls_override = config.pop("type", None)
    if isinstance(cls_override, type):
        cls = cls_override
    elif isinstance(cls_override, str):
        try:
            cls = get_scope()[cls_override]
        except KeyError as e:
            cls = type(cls_override, (Directory,), {})
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        "Casting to a new subclass of Directory because "
                        f'"{cls_override}" can\'t be resolved as it is not found'
                        + f" inside the current scope of Directory subclasses."
                        + " This dynamically created subclass allows to view the data"
                        + " without access to the original class definition and methods."
                        + " If this happens unexpectedly with autoreload enabled in"
                        + " a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)`"
                        + " as a workaround or restart the kernel"
                        + f" (background: https://github.com/ipython/ipython/issues/12399)."
                    ),
                    ConfigWarning,
                    stacklevel=2,
                )
            # raise KeyError(
            #     f'"{cls_override}" can\'t be resolved because it is not found'
            #     + f" inside the current scope of Directory subclasses."
            #     + " If this happens unexpectedly with autoreload enabled in"
            #     + " a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)`"
            #     + " as a workaround or restart the kernel"
            #     + f" (background: https://github.com/ipython/ipython/issues/12399)."
            # ) from e

    # Construct and return a Directory instance
    obj = object.__new__(cls)
    default_config = get_defaults(cls)
    default_config.update(config)
    config = Namespace(type=_identify(type(obj)), **default_config)
    object.__setattr__(obj, "_config", namespacify(config))
    return cast(Directory, obj)


# -- I/O -----------------------------------------------------------------------


def _read_h5(path: Path, assert_swmr=True) -> ArrayFile:
    try:
        f = h5.File(path, "r", libver="latest", swmr=True)
        if assert_swmr:
            assert f.swmr_mode, "File is not in SWMR mode."
        return f["data"]
    except OSError as e:
        print(e)
        if "errno = 2" in str(e):
            raise e
        sleep(0.1)
        return _read_h5(path)

    # try:
    #     f = h5.File(path, "r", libver="latest", swmr=True)
    #     assert f.swmr_mode
    #     return f["data"]
    # except OSError as e:
    #     print(e)
    #     if "errno = 2" in str(e):  # 2 := File not found.
    #         raise e
    #     sleep(0.1)
    #     return _read_h5(path)


def _write_h5(path: Path, val: object) -> None:
    val = np.asarray(val)
    try:
        f = h5.File(path, libver="latest", mode="w")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        f = h5.File(path, libver="latest", mode="w")
        f["data"] = val
        f.swmr_mode = True
        assert f.swmr_mode


def _extend_h5(path: Path, val: object, retry: int = 0, max_retries: int = 50) -> None:
    val = np.asarray(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mode='a' to read file, create otherwise
    try:
        f = h5.File(path, libver="latest", mode="a")
        if "data" not in f:
            dset = f.require_dataset(
                name="data",
                shape=None,
                maxshape=(None, *val.shape[1:]),
                dtype=val.dtype,
                data=np.empty((0, *val.shape[1:]), val.dtype),
                chunks=(
                    int(np.ceil(2**12 / np.prod(val.shape[1:]))),
                    *val.shape[1:],
                ),
            )
            f.swmr_mode = True
        else:
            dset = f["data"]
    except BlockingIOError as e:
        print(e)
        if "errno = 11" in str(e) or "errno = 35" in str(
            e
        ):  # 11, 35 := Reource temporarily unavailable
            sleep(0.1)
            if retry < max_retries:
                _extend_h5(path, val, retry + 1, max_retries)
            else:
                raise RecursionError(
                    "maximum retries to extend the dataset"
                    " exceeded, while the resource was unavailable. Because"
                    " the dataset is constantly locked by another thread."
                )
            return
        else:
            raise e

    def _override_to_chunked(path: Path, val: object) -> None:
        # override as chunked dataset
        data = _read_h5(path, assert_swmr=False)[()]
        path.unlink()
        _extend_h5(path, data)
        # call extend again with new value
        _extend_h5(path, val)

    if len(val) > 0:
        try:
            dset.resize(dset.len() + len(val), 0)
            dset[-len(val) :] = val
            dset.flush()
        except TypeError as e:
            # workaround if dataset was first created as non-chunked
            # using __setitem__ and then extended using extend
            if "Only chunked datasets can be resized" in str(e):
                _override_to_chunked(path, val)
            else:
                raise e


def _copy_file(dst: Path, src: Path) -> None:
    # shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _copy_dir(dst: Path, src: Path) -> None:
    # shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _copy_if_conflict(src):
    _existing_conflicts = src.parent.glob(f"{src.name}_conflict*")
    max_count = max([int(c.name[-4:]) for c in _existing_conflicts])
    dst = src.parent / f"{src.name}_conflict_{max_count+1:04}"
    shutil.copytree(src, dst)
    return dst


def _extend_file(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_src:
        with open(dst, "ab+") as f_dst:
            f_dst.write(f_src.read())


def read_meta(path: Path) -> Namespace:
    # TODO: Implement caching
    try:
        # meta = namespacify(yaml.safe_load((path/'_meta.yaml').read_text()))
        meta = namespacify(json.loads((path / "_meta.yaml").read_text()))
        assert isinstance(meta, Namespace)
        if hasattr(meta, "config"):
            assert isinstance(meta.config, Namespace)
        elif hasattr(meta, "spec"):  # for backwards compatibility
            assert isinstance(meta.spec, Namespace)
            meta["config"] = meta.pop("spec")
        assert isinstance(meta.status, str)
        return meta
    except:
        return Namespace(config=None, status="done")


def directory_to_dict(directory: Directory) -> dict:
    dw_dict = {
        key: getattr(directory, key)[...]
        for key in list(directory.keys())
        if isinstance(getattr(directory, key), h5.Dataset)
    }
    return dw_dict


def directory_to_df(directory: Directory, dtypes: dict = None) -> DataFrame:
    """Convert a directory to a pandas DataFrame."""
    df_dict = {
        key: getattr(directory, key)[...]
        for key in list(directory.keys())
        if isinstance(getattr(directory, key), h5.Dataset)
    }

    # Get the lengths of all datasets.
    nelements = {k: len(v) or 1 for k, v in df_dict.items()}

    lengths, counts = np.unique([val for val in nelements.values()], return_counts=True)
    most_frequent_length = lengths[np.argmax(counts)]

    # If there are single element datasets, just create a new column of most_frequent_length and put the value in each row.
    if lengths.min() == 1:
        for k, v in nelements.items():
            if v == 1:
                df_dict[k] = df_dict[k].repeat(most_frequent_length)

    df_dict = byte_to_str(df_dict)

    if dtypes is not None:
        df_dict = {
            k: np.array(v).astype(dtypes[k]) for k, v in df_dict.items() if k in dtypes
        }
    return DataFrame.from_dict(
        {k: v.tolist() if v.ndim > 1 else v for k, v in df_dict.items()}
    )


def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
    last_modified=False,
    not_exists_msg="path does not exist",
    verbose=True,
):
    """Given a directory Path object print a visual tree structure"""
    # prefix components:
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "

    tree_string = ""

    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 1

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories
        if not level:
            yield prefix + "..."
            return  # 0, stop iterating
        if limit_to_directories:
            contents = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        else:
            contents = sorted(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name + "/"
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    tree_string += dir_path.name + "/"

    if not dir_path.exists():
        tree_string += f"\n{space}({not_exists_msg})"
        return tree_string

    if last_modified:
        timestamp = datetime.datetime.fromtimestamp(dir_path.stat().st_mtime)
        mtime = " - Last modified: {}".format(timestamp.strftime("%B %d, %Y %H:%M:%S"))
        tree_string += mtime
    tree_string += "\n"
    iterator = inner(dir_path, level=level)
    for line in itertools.islice(iterator, length_limit):
        tree_string += line + "\n"

    if verbose:
        if next(iterator, None):
            tree_string += f"... length_limit, {length_limit}, reached,"
        tree_string += (
            f"\ndisplaying: {directories} {'directory' if directories == 1 else 'directories'}"
            + (f", {files} files" if files else "")
            + (f", {level} levels." if level >= 1 else "")
        )

    return tree_string


def byte_to_str(obj):
    """Cast byte elements to string types.

    Note, this function is recursive and will cast all byte elements in a nested
    list or tuple.
    """
    if isinstance(obj, Mapping):
        return type(obj)({k: byte_to_str(v) for k, v in obj.items()})
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    elif isinstance(obj, list):
        obj = [byte_to_str(item) for item in obj]
        return obj
    elif isinstance(obj, tuple):
        obj = tuple([byte_to_str(item) for item in obj])
        return obj
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError(f"can't cast {obj} of type {type(obj)} to str")


# -- Scope search --------------------------------------------------------------


def _identify(type_: type) -> str:
    for sym, t in get_scope().items():
        # comparing t == type_ can yield false in combination with
        # ipython autoreload, therefore relying on comparing the __qualname__
        if t.__qualname__ == type_.__qualname__:
            return sym


def _identify_elements(obj: object) -> object:
    if isinstance(obj, type):
        return _identify(obj)
    elif isinstance(obj, list):
        return [_identify_elements(elem) for elem in obj]
    elif isinstance(obj, dict):
        return Namespace({k: _identify_elements(obj[k]) for k in obj})
    else:
        return obj
