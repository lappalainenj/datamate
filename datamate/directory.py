"""
This module exports the `Directory` class, an array- and metadata-friendly view
into a directory.

Instances of the base Directory class have methods to simplify reading/writing
collections of arrays. `Directory` can also be subclassed to define Configurable,
persistent computed asset types, within Python/PEP 484's type system.

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
)
from typing_extensions import Protocol
import datetime
from traceback import format_tb


from contextlib import contextmanager

import h5py as h5
import numpy as np
from pandas import DataFrame
from ruamel import yaml
from toolz import valmap

from datamate.namespaces import Namespace, namespacify

__all__ = ["Directory", "ArrayFile"]

# -- Custom Errors and Warnings ------------------------------------------------


class ConfigWarning(Warning):
    pass


class ModifiedWarning(Warning):
    pass


class ModifiedError(Exception):
    pass


# -- Static type definitions ---------------------------------------------------

from pathlib import Path as Path


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
    """Wraps a callable to control its root directory at execution.

    root_dir (str or Path): root directory that will be set at execution of the
        callable. Optional. Default is None which corresponds to get_root_dir().

    Note, this will take precedence over set_root_dir, and set_root_context context when
    the callable is called within the context.
    """

    def decorator(callable):
        @functools.wraps(callable)
        def function(*args, **kwargs):
            _root_dir = get_root_dir()
            set_root_dir(root_dir or _root_dir)
            _return = callable(*args, **kwargs)
            set_root_dir(_root_dir)
            return _return

        return function

    return decorator


@contextmanager
def set_root_context(root_dir: Union[str, Path, NoneType] = None):
    """Set root directory within a context and revert after.

    Example:
        with set_root_context(dir):
            Directory(config)
    """
    _root_dir = get_root_dir()
    set_root_dir(root_dir)
    yield
    set_root_dir(_root_dir)


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


def set_verbosity_level(level: int) -> None:
    """
    Set verbosity level of representation for Directorys.

    0: only the top level directory name and the last modified date are shown.
    1: maximally 2 levels and 25 lines are represented.
    2: all directorys and files are represented.

    Defaults to 2.
    """
    context.verbosity_level = level


def get_check_size_on_init() -> bool:
    return context.check_size_on_init


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
        path (Path|str): The path at which the Directory is, or should be,
            stored
        conf (Mapping[str, object]): The build Namespace, optionally
            including a "type" field indicating the type of Directory to search
            for/construct

    Constructors:
         - Directory(name: str)\n'
         - Directory(name: str, conf: Mapping[str, object])'
         - Directory(conf: Mapping[str, object])\n'
         - Directory(path: Path|str)\n'
         - Directory(path: Path|str, conf: Mapping[str, object])\n'


    If only `path` is provided, the Directory corresponding to `Path` is
    returned. It will be empty if `path` points to an empty or nonexistent
    directory.

    If only `conf` is provided, it will search the current `root_dir`
    for a matching directory, and return a Directory pointing there if it
    exists. Otherwise, a new Directory will be constructed at the top level of
    the `root_dir`.

    If both `path` and `conf` are provided, it will return the Directory
    at `path`, building it if necessary. If `path` points to an existing
    directory that is not a sucessfully built Directory matching `conf`, an
    error is raised.

    Fields:
        - **path** (*Path*): The path to the root of the file tree backing this \
            Directory
        - **conf** (*Namespace*): The Namespace (inherited from
            `Configurable`)
        - **meta** (*Namespace*): The metadata stored in \
            `{self.path}/_meta.yaml`

    After instantiation, Directorys act as string-keyed `MutableMapping`s (with
    some additional capabilities), containing three types of entries:
    `ArrayFile`s, `Path`s, and other `Directory`s.

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
    Directorys or (possibly nested) string-keyed `Mapping`s (*e.g* a dictionary
    of arrays).
    """

    class Config(Protocol):
        """
        A configuration

        If its definition is inline (lexically within the containing class'
        definition), it will be translated into a JSON-Schema to validate
        configurations passed into the outer class' constructor.

        `Config` classes are intended to be interface definitions. They can extend
        `typing_extensions.Protocol` to support static analysis.

        An empty `Conf` definition is created for every `Configurable` subclass
        defined without one.
        """

        pass

    path: Path
    config: Config

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        path, config = _parse_directory_args(args, kwargs)
        if path is None and config is None:
            return _directory(cls)
        elif path is not None and config is None:
            return _check_size(_directory_from_path(cls, _resolve_path(path)))
        elif path is None and config is not None:
            return _check_size(_directory_from_config(cls, config))
        elif path is not None and config is not None:
            return _check_size(
                _directory_from_path_and_config(cls, _resolve_path(path), config)
            )

    def __init__(self, config: Config):
        """Implement to compile data at runtime from a configuration."""
        pass

    @property
    def meta(self) -> Namespace:
        """
        The metadata stored in `{self.path}/_meta.yaml`
        """
        return read_meta(self.path)

    @property
    def config(self):
        return self.meta.config

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

    def size(self) -> None:
        __check_size = get_check_size_on_init()
        check_size_on_init(True)
        _check_size(self, print_size=True)
        check_size_on_init(__check_size)

    def items(self) -> Iterator[Tuple[str, ArrayFile]]:
        for key in self.keys():
            yield (key, self[key])

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

    @classmethod
    def from_df(cls, df: DataFrame, dtypes: dict, *args, **kwargs):
        directory = Directory.__new__(Directory, *args, **kwargs)
        directory.update(
            {column: df[column].values.astype(dtypes[column]) for column in df.columns}
        )
        return directory

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

    # TODO: from dict

    def update(self, other, suffix: str = "") -> None:
        """
        Updates self with items of other and appends an optional suffix.
        """
        for key in other:
            if key + suffix not in self:
                self[key + suffix] = other[key]

    def move(self, dst):
        """Move directory to dst."""
        return shutil.move(self.path, dst)

    def rmtree(self, y_n=None):
        reply = y_n or input(f"delete {self.path} recursively, y/n?")
        if reply.lower() == "y":
            shutil.rmtree(self.path, ignore_errors=True)

    def _rebuild(self, y_n=None):
        self.rmtree(y_n)
        _build(self)

    def check_size(self, print=True):
        return check_size(self.path, print_size=print)

    def mtime(self):
        return datetime.datetime.fromtimestamp(self.path.stat().st_mtime)

    @property
    def parent(self):
        return Directory(self.path.absolute().parent)

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
        try:
            # to catch cases where key is an index to a reference to an h5 file.
            # this will yield a TypeError because Path / slice does not work.
            path = self.path / key
        except TypeError as e:
            if not self.path.exists():
                raise (
                    AssertionError(
                        f"Indexing {self.path.name} at {key} not possible for"
                        f"Directory at {self.path.parent}. File "
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

    def extend(self, key: str, val: object) -> None:
        """
        Extends an `ArrayFile`, `Path`, or `Directory` at `self.path/key`

        Extending `ArrayFile`s performs concatenation along the first axis,
        extending `Path`s performs byte-level concatenation, and
        extending subDirectorys extends their fields.

        Files corresponding to `self[key]` are created if they do not already
        exist.
        """
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

    # -- Other convenience methods --------------------

    def _clear_dir(self, suffix: str) -> None:
        """
        Delete files ending with suffix in the current wrap path
        """
        for file in self.path.iterdir():
            if file.is_file() and file.suffix == suffix:
                file.unlink()


# -- Directory construction -----------------------------------------------------


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
            "    - Directory(conf: Mapping[str, object])\n"
            "    - Directory(path: Path|str)\n"
            "    - Directory(path: Path|str, conf: Mapping[str, object])\n"
            "    - Directory(name: str)\n"
            "    - Directory(name: str, conf: Mapping[str, object])"
        )


def _directory(cls: type) -> Directory:
    """
    Return a new Directory corresponding to the file tree at root_dir / cls.__name___ *
    """
    directory = _forward_subclass(cls, {})
    path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _check_for_init(cls: type) -> bool:
    # import pdb; pdb.set_trace()
    return inspect.getsource(cls.__init__).split("\n")[-2].replace(" ", "") != "pass"


def _directory_from_path(cls: type, path: Path) -> Directory:
    """
    Return a Directory corresponding to the file tree at `path`.

    An error is raised if the type recorded in `_meta.yaml`, if any, is not a
    subtype of `cls`.
    """
    config = read_meta(path).config or {}
    written_type = get_scope().get(config.get("type", None), None)

    if path.is_file():
        raise FileExistsError(f"{path} is a file.")

    if context.enforce_config_match:
        if _check_for_init(cls) and not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist.")

        if written_type is not None and not issubclass(written_type, cls):
            raise FileExistsError(
                f"{path} is a {written_type.__module__}.{written_type.__qualname__}"
                f", not a {cls.__module__}.{cls.__qualname__}."
            )

    if context.enforce_config_match:
        directory = _forward_subclass(cls, config)
    else:
        directory = _forward_subclass(cls, {})
    object.__setattr__(directory, "_cached_keys", set())
    object.__setattr__(directory, "path", path)
    return directory


def _directory_from_config(cls: type, conf: Mapping[str, object]) -> Directory:
    """
    Find or build a Directory with the given type and Namespace.
    """
    directory = _forward_subclass(cls, conf)
    new_dir_path = _new_directory_path(type(directory))
    object.__setattr__(directory, "_cached_keys", set())
    config = Namespace(**directory._config)

    def _new_directory():
        object.__setattr__(directory, "path", new_dir_path)
        # return empty path cause only the type field is populated
        if list(config.keys()) == ["type"]:
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
    cls: type, path: Path, conf: Mapping[str, object]
) -> Directory:
    """
    Find or build a Directory with the given type, path, and Namespace.
    """
    directory = _forward_subclass(cls, conf)
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
    # TODO: Fix YAML generation.

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
            build_args = [directory._config] if n_build_args > 1 else []
            directory.__init__(*build_args)
        write_meta(config=config, status="done")
    except BaseException as e:
        write_meta(config=config, status="stopped")
        raise e


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


def _check_size(directory: Directory, warning_at=20 * 1024**3, print_size=False):
    """
    Checks the size of the Directory directory on instantiation and warns if it exceeds 20GiB (default).
    """
    if context.check_size_on_init:
        _check_size(directory.path, warning_at, print_size)
    return directory


def check_size(path: Path, warning_at=20 * 1024**3, print_size=False) -> None:
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
            raise KeyError(
                f'"{cls_override}" can\'t be resolved because it is not found'
                + f" inside the scope of Directory subclasses. Typo?"
                + " If this happens in the context of autoreload enabled in"
                + " a notebook/IPython session, run `datamate.reset_scope(datamate.Directory)`"
                + " as a workaround or restart the kernel"
                + f" (background: https://github.com/ipython/ipython/issues/12399)."
            ) from e

    # Construct and return a `Configurable` instance.
    obj = object.__new__(cls)
    config = Namespace(type=_identify(type(obj)), **config)
    object.__setattr__(obj, "_config", namespacify(config))
    return cast(Directory, obj)


# -- I/O -----------------------------------------------------------------------


def _read_h5(path: Path) -> ArrayFile:
    try:
        f = h5.File(path, "r", libver="latest", swmr=True)
        assert f.swmr_mode
        return f["data"]
    except OSError as e:
        print(e)
        if "errno = 2" in str(e):  # 2 := File not found.
            raise e
        sleep(0.1)
        return _read_h5(path)


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
    except BlockingIOError as e:
        print(e)
        if "errno = 11" in str(e):  # 11 := Reource temporarily unavailable
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
    if len(val) > 0:
        dset.resize(dset.len() + len(val), 0)
        dset[-len(val) :] = val
        dset.flush()


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
    def convert(_arr):
        if isinstance(_arr, np.ndarray):
            if isinstance(_arr.item(0), (np.character, bytes)):
                return _arr.astype(str)
        return _arr

    df_dict = {
        key: getattr(directory, key)[...]
        for key in list(directory.keys())
        if isinstance(getattr(directory, key), h5.Dataset)
    }

    # Get the lengths of all datasets.
    _lengths = {k: len(v) or 1 for k, v in df_dict.items()}

    lengths, counts = np.unique([val for val in _lengths.values()], return_counts=True)
    most_frequent_length = lengths[np.argmax(counts)]

    # If there are single element datasets, just create a new column of most_frequent_length and put the value in each row.
    if lengths.min() == 1:
        for k, v in _lengths.items():
            if v == 1:
                df_dict[k] = df_dict[k].repeat(most_frequent_length)

    df_dict = valmap(
        convert,
        {key: val for key, val in df_dict.items() if len(val) == most_frequent_length},
    )
    if dtypes is not None:
        df_dict = {
            k: np.array(v).astype(dtypes[k]) for k, v in df_dict.items() if k in dtypes
        }

    return DataFrame.from_dict(df_dict)


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
