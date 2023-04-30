from pathlib import Path
from typing import List

import h5py as h5
import numpy as np
import pytest
import shutil

from datamate import (
    Directory,
    Namespace,
    set_root_dir,
    get_root_dir,
    root,
    set_root_context,
)
from datamate.directory import (
    ModifiedError,
    ModifiedWarning,
    ConfigWarning,
    ImplementationWarning,
    ImplementationError,
)

# -- Helper functions ----------------------------------------------------------


def data_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.random.rand(3).tobytes())
    return path


def data_file_concat(path: Path, args: List[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(a.read_bytes() for a in args))
    return path


def assert_directory_equals(directory: Directory, target: dict) -> None:
    if "__type__" in target:
        assert isinstance(directory, target.pop("__type__"))
    if "__path__" in target:
        assert directory.path == target.pop("__path__")
    if "__conf__" in target:
        assert directory._config == target.pop("__conf__")
    if "__meta__" in target:
        assert directory.meta == target.pop("__meta__")

    assert len(directory) == len(target)
    assert len(list(directory)) == len(target)

    for k, v in target.items():
        assert k in directory
        assert k in list(directory)
        assert hasattr(directory, k)

        if isinstance(v, dict):
            assert isinstance(directory[k], Directory)
            assert isinstance(getattr(directory, k), Directory)
            assert_directory_equals(directory[k], v)
            assert_directory_equals(getattr(directory, k), v)

        elif isinstance(v, Path):
            k_mod = k.replace(".", "__")
            assert (directory.path / k).is_file()
            assert isinstance(directory[k], Path)
            assert isinstance(getattr(directory, k), Path)
            assert directory[k].read_bytes() == v.read_bytes()
            assert getattr(directory, k).read_bytes() == v.read_bytes()
            assert getattr(directory, k_mod).read_bytes() == v.read_bytes()

        else:
            assert (directory.path / k).with_suffix(".h5").is_file()
            assert isinstance(directory[k], h5.Dataset)
            assert isinstance(getattr(directory, k), h5.Dataset)
            assert np.array_equal(directory[k][()], v)
            assert np.array_equal(getattr(directory, k)[()], v)
            assert directory[k].dtype == np.asarray(v).dtype
            assert getattr(directory, k).dtype == np.asarray(v).dtype


# -- [Base class tests] Empty directories --------------------------------------


def test_empty_directory_with_existing_dir(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    assert_directory_equals(
        a,
        dict(
            __path__=tmp_path,
            __conf__=Namespace(type="Directory"),
            __meta__={"config": None, "status": "done"},
        ),
    )
    assert isinstance(a.nonexistent_entry, Directory)


def test_empty_directory_with_nonexistent_dir(tmp_path: Path) -> None:
    a = Directory(tmp_path / "new_dir")
    assert_directory_equals(
        a,
        dict(
            __path__=tmp_path / "new_dir",
            __conf__=Namespace(type="Directory"),
            __meta__={"config": None, "status": "done"},
        ),
    )
    assert isinstance(a.nonexistent_entry, Directory)
    assert not (tmp_path / "new_dir").exists()


# -- [Base class tests] Entry assignment ---------------------------------------


def test_float_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.b = 1.5
    a.b = 2.5
    a["c"] = 3.5
    a["c"] = 4.5
    assert_directory_equals(a, {"b": 2.5, "c": 4.5})


def test_byte_string_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.b = b"bee"
    a.b = b"buzz"
    a["c"] = b"sea"
    a["c"] = b"ahoy!"
    assert_directory_equals(a, {"b": b"buzz", "c": b"ahoy!"})


def test_list_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.b = [1, 2, 3]
    a.b = [4, 5, 6]
    a["c"] = [[7, 8], [9, 10]]
    a["c"] = [[11, 12, 13], [14, 15, 16]]
    assert_directory_equals(a, {"b": [4, 5, 6], "c": [[11, 12, 13], [14, 15, 16]]})


def test_array_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.b = np.ones((2, 3), dtype="float32")
    a.b = np.zeros((4, 5, 6), dtype="float32")
    a["c"] = np.ones((4, 5, 6), dtype="float32")
    a["c"] = np.ones((2, 3), dtype="uint16")
    assert_directory_equals(
        a,
        {
            "b": np.zeros((4, 5, 6), dtype="float32"),
            "c": np.ones((2, 3), dtype="uint16"),
        },
    )


def test_path_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path / "directory")
    a.b__bin = data_file(tmp_path / "b0.bin")
    a.b__bin = data_file(tmp_path / "b1.bin")
    a["c.bin"] = data_file(tmp_path / "c0.bin")
    a["c.bin"] = data_file(tmp_path / "c1.bin")
    assert_directory_equals(
        a, {"b.bin": tmp_path / "b1.bin", "c.bin": tmp_path / "c1.bin"}
    )


def test_dict_assignment(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.dict = dict(a=[1, 2, 3], b=dict(c=[4], d=[5, 6]))
    assert_directory_equals(a, {"dict": {"a": [1, 2, 3], "b": {"c": [4], "d": [5, 6]}}})


def test_directory_assignment(tmp_path: Path) -> None:
    a_src = Directory(tmp_path / "a_src")
    a_src.b.c = b"bee sea"
    a_src.d.e = [1, 2, 3, 4]
    a_src.f.g__bin = data_file(tmp_path / "effigy.bin")
    a_dst = Directory(tmp_path / "a_dst")
    a_dst.a = a_src
    assert_directory_equals(
        a_dst,
        {
            "a": {
                "b": {"c": b"bee sea"},
                "d": {"e": [1, 2, 3, 4]},
                "f": {"g.bin": tmp_path / "effigy.bin"},
            }
        },
    )


# -- [Base class tests] Entry extension ----------------------------------------


def test_list_extension(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.extend("b", [[7, 8], [9, 10]])
    a.extend("b", [[11, 12]])
    assert_directory_equals(a, {"b": [[7, 8], [9, 10], [11, 12]]})


def test_array_extension(tmp_path: Path) -> None:
    a = Directory(tmp_path)
    a.extend("b", np.uint16([[7, 8], [9, 10]]))
    a.extend("b", np.uint16([[11, 12]]))
    assert_directory_equals(a, {"b": np.uint16([[7, 8], [9, 10], [11, 12]])})


def test_path_extension(tmp_path: Path) -> None:
    a = Directory(tmp_path / "a")
    a.extend("b.bin", data_file(tmp_path / "b0.bin"))
    a.extend("b.bin", data_file(tmp_path / "b1.bin"))
    assert_directory_equals(
        a,
        {
            "b.bin": data_file_concat(
                tmp_path / "b2.bin", [tmp_path / "b0.bin", tmp_path / "b1.bin"]
            )
        },
    )


def test_dict_extension(tmp_path: Path) -> None:
    a = Directory(tmp_path / "a")
    a.extend(
        "b",
        {
            "c": np.empty((0, 2), dtype="uint16"),
            "d.bin": data_file(tmp_path / "d0.bin"),
            "e": {"f": [0.1, 0.2]},
        },
    )
    a.extend(
        "b",
        {
            "c": np.uint16([[1, 2], [3, 4]]),
            "d.bin": data_file(tmp_path / "d1.bin"),
            "e": {"f": [0.3, 0.4, 0.5]},
        },
    )
    assert_directory_equals(
        a,
        {
            "b": {
                "c": np.uint16([[1, 2], [3, 4]]),
                "d.bin": data_file_concat(
                    tmp_path / "d2.bin", [tmp_path / "d0.bin", tmp_path / "d1.bin"]
                ),
                "e": {"f": [0.1, 0.2, 0.3, 0.4, 0.5]},
            }
        },
    )


def test_directory_extension(tmp_path: Path) -> None:
    a0 = Directory(tmp_path / "a0")
    a0.b = [b"hello"]
    a0.c__bin = data_file(tmp_path / "c0.bin")
    a0.d = {"e": [[0.0, 1.0]], "f.bin": data_file(tmp_path / "f0.bin")}

    a1 = Directory(tmp_path / "a1")
    a1.b = [b"good", b"bye"]
    a1.c__bin = data_file(tmp_path / "c1.bin")
    a1.d = {"e": np.empty((0, 2)), "f.bin": data_file(tmp_path / "f1.bin")}

    a2 = Directory(tmp_path / "a2")
    a2.extend("subdirectory", a0)
    a2.extend("subdirectory", a1)

    assert_directory_equals(
        a2,
        {
            "subdirectory": {
                "b": [b"hello", b"good", b"bye"],
                "c.bin": data_file_concat(
                    tmp_path / "c2.bin", [tmp_path / "c0.bin", tmp_path / "c1.bin"]
                ),
                "d": {
                    "e": [[0.0, 1.0]],
                    "f.bin": data_file_concat(
                        tmp_path / "f2.bin", [tmp_path / "f0.bin", tmp_path / "f1.bin"]
                    ),
                },
            }
        },
    )


# -- [Base class tests] Entry deletion -----------------------------------------


def test_array_file_deletion(tmp_path: Path) -> None:
    a = Directory(tmp_path / "a")
    a.b = [1, 2, 3]
    a.c = b"four five six"
    a.d = [7, 8]
    a.e = {"blue": b"jeans"}
    a.f__bin = data_file(tmp_path / "data.bin")
    del a.b
    del a["c"]
    assert_directory_equals(
        a, {"d": [7, 8], "e": {"blue": b"jeans"}, "f.bin": tmp_path / "data.bin"}
    )


def test_opaque_file_deletion(tmp_path: Path) -> None:
    a = Directory(tmp_path / "a")
    a.b__bin = data_file(tmp_path / "b.bin")
    a.c__bin = data_file(tmp_path / "c.bin")
    a.d = [7, 8]
    a.e = {"blue": b"jeans"}
    a.f__bin = data_file(tmp_path / "data.bin")
    del a.b__bin
    del a["c.bin"]
    assert_directory_equals(
        a, {"d": [7, 8], "e": {"blue": b"jeans"}, "f.bin": tmp_path / "data.bin"}
    )


def test_directory_deletion(tmp_path: Path) -> None:
    a = Directory(tmp_path / "a")
    a.b = {"aa": {"bb": 0, "cc": 1}}
    a.c = {"dd": {"ee": [2, 3, 4]}}
    a.d = [7, 8]
    a.e = {"blue": b"jeans"}
    a.f__bin = data_file(tmp_path / "data.bin")
    del a.b
    del a["c"]
    assert_directory_equals(
        a, {"d": [7, 8], "e": {"blue": b"jeans"}, "f.bin": tmp_path / "data.bin"}
    )


# -- [Subclass tests] Construction ---------------------------------------------


class CustomDirectory(Directory):
    n_calls = 0

    class Config:
        n_zeros: int
        n_ones: int

    def __init__(self, conf) -> None:
        CustomDirectory.n_calls += 1
        self.zeros = np.zeros(conf.n_zeros)
        self.ones = np.ones(conf.n_ones)


class AnotherDirectory(Directory):
    def __init__(self) -> None:
        AnotherDirectory.Config.n_calls += 1


def test_construction_from_nothing(tmp_path: Path) -> None:
    # Setup
    set_root_dir(tmp_path)
    AnotherDirectory.Config.n_calls = 0

    # Case 1: (not exists)
    a0 = AnotherDirectory()
    assert AnotherDirectory.Config.n_calls == 0

    # Case 2: (exists, empty)
    path = a0.path
    a1 = AnotherDirectory()
    assert path == a1.path

    # Case 3: (exists, non-empty)
    a0.__init__()
    a1 = AnotherDirectory()
    assert AnotherDirectory.Config.n_calls == 1

    # Cleanup
    set_root_dir(Path("."))


def test_construction_from_path(tmp_path: Path) -> None:
    # Setup
    set_root_dir(tmp_path)
    CustomDirectory.n_calls = 0
    a0 = CustomDirectory(n_zeros=2, n_ones=3)

    # Case 1: (path_given, exists)
    a1 = CustomDirectory(a0.path)
    a2 = CustomDirectory(f"{a0.path}")
    a3 = CustomDirectory(f"@/{a0.path.name}")
    assert_directory_equals(a1, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert_directory_equals(a2, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert_directory_equals(a3, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert CustomDirectory.n_calls == 1

    # Case 2: (path_given, does_not_exists)
    with pytest.raises(FileNotFoundError):
        CustomDirectory(tmp_path / "invalid_path")

    # Cleanup
    set_root_dir(Path("."))


def test_construction_from_conf(tmp_path: Path) -> None:
    # Setup
    set_root_dir(tmp_path)
    CustomDirectory.n_calls = 0
    a0 = CustomDirectory(n_zeros=2, n_ones=3)

    # Case 1: (conf_given, exists)
    a1 = CustomDirectory(n_zeros=2, n_ones=3)
    a2 = CustomDirectory(dict(n_zeros=2, n_ones=3))
    assert_directory_equals(a1, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert_directory_equals(a2, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert a1.path == a0.path
    assert a2.path == a0.path
    assert CustomDirectory.n_calls == 1

    # Case 2: (conf_given, does_not_exist)
    a3 = CustomDirectory(n_zeros=2, n_ones=4)
    a4 = CustomDirectory(dict(n_zeros=1, n_ones=3))
    assert_directory_equals(a3, {"zeros": np.zeros(2), "ones": np.ones(4)})
    assert_directory_equals(a4, {"zeros": np.zeros(1), "ones": np.ones(3)})
    assert CustomDirectory.n_calls == 3

    # Cleanup
    set_root_dir(Path("."))


def test_construction_from_path_and_conf(tmp_path: Path) -> None:
    # Setup
    set_root_dir(tmp_path)
    CustomDirectory.n_calls = 0
    a0 = CustomDirectory(n_zeros=2, n_ones=3)

    # Case 1: (path_given, conf_given, exists_and_matches)
    a1 = CustomDirectory(a0.path, n_zeros=2, n_ones=3)
    a2 = CustomDirectory(a0.path, dict(n_zeros=2, n_ones=3))
    a3 = CustomDirectory(f"@/{a0.path.name}", n_zeros=2, n_ones=3)
    assert_directory_equals(a1, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert_directory_equals(a2, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert_directory_equals(a3, {"zeros": np.zeros(2), "ones": np.ones(3)})
    assert a1.path == a0.path
    assert a2.path == a0.path
    assert a3.path == a0.path
    assert CustomDirectory.n_calls == 1

    # Case 2: (path_given, conf_given, exists_and_does_not_match)
    with pytest.raises(FileExistsError):
        CustomDirectory(a0.path, n_zeros=1, n_ones=3)
    with pytest.raises(FileExistsError):
        CustomDirectory(a0.path, dict(n_zeros=1, n_ones=4))
    with pytest.raises(FileExistsError):
        CustomDirectory(f"@/{a0.path.name}", n_zeros=2, n_ones=4)
    assert CustomDirectory.n_calls == 1

    # Case 3: (path_given, conf_given, does_not_exist)
    a4 = CustomDirectory(tmp_path / "a4", n_zeros=1, n_ones=3)
    a5 = CustomDirectory(tmp_path / "a5", dict(n_zeros=1, n_ones=4))
    a6 = CustomDirectory("@/a6", n_zeros=2, n_ones=4)
    assert_directory_equals(a4, {"zeros": np.zeros(1), "ones": np.ones(3)})
    assert_directory_equals(a5, {"zeros": np.zeros(1), "ones": np.ones(4)})
    assert_directory_equals(a6, {"zeros": np.zeros(2), "ones": np.ones(4)})
    assert a4.path == tmp_path / "a4"
    assert a5.path == tmp_path / "a5"
    assert a6.path == tmp_path / "a6"
    assert CustomDirectory.n_calls == 4

    # Cleanup
    set_root_dir(Path("."))


def test_modified_error(tmp_path: Path) -> None:
    # Setup
    set_root_dir(tmp_path)
    CustomDirectory.n_calls = 0
    a0 = CustomDirectory(n_zeros=2, n_ones=3)

    # modify a0
    a0.zeros = np.ones(3)

    assert a0.meta.modified

    with pytest.warns(ModifiedWarning):
        CustomDirectory(n_zeros=2, n_ones=3)


# -- [Subclass tests] Build customization --------------------------------------


class DirectoryWithUnaryBuild(Directory):
    class Config:
        prop: int

    def __init__(self) -> None:
        self.field = self.config.prop


class DirectoryWithBinaryBuild(Directory):
    class Config:
        prop: int

    def __init__(self, conf) -> None:
        self.field = conf.prop


def test_build_customization(tmp_path: Path) -> None:
    a_unary = DirectoryWithUnaryBuild(tmp_path / "unary", prop=10)
    a_binary = DirectoryWithBinaryBuild(tmp_path / "binary", prop=10)
    assert a_unary.field[()] == 10
    assert a_binary.field[()] == 10


# -- Test root control ---------------------------------------------------------


@pytest.fixture(scope="session")
def rooted_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("rooted_dir")

    @root(path)
    class RootedDirectory(Directory):
        class Config:
            start: int
            stop: int
            step: int

        def __init__(self, config) -> None:
            self.array = np.arange(config.start, config.stop, config.step)

    return RootedDirectory, path


def test_root_dir_provided(tmp_path, rooted_dir):
    RootedDirectory, rooted_dir_root_path = rooted_dir

    assert rooted_dir_root_path != tmp_path

    # case 1: root dir provided in decorator
    dir = RootedDirectory(Namespace(start=0, stop=10, step=1))
    assert dir.path.parent == rooted_dir_root_path

    # case 2: root dir provided and not within context
    set_root_dir(tmp_path)
    dir = RootedDirectory(Namespace(start=0, stop=10, step=1))
    assert dir.path.parent == rooted_dir_root_path

    # case 3: root_dir provided and within context
    with set_root_context(tmp_path):
        dir = RootedDirectory(Namespace(start=0, stop=10, step=1))
        assert dir.path.parent == tmp_path


@root()
class RootedDirectory(Directory):
    def __init__(self, config) -> None:
        self.array = np.arange(config.start, config.stop, config.step)


def test_root_dir_not_provided(tmp_path):
    set_root_dir(tmp_path)

    # case 4: root dir not provided and not within context
    dir = RootedDirectory(Namespace(start=0, stop=10, step=1))
    assert dir.path.parent == tmp_path

    # case 5: root dir not provided and within context
    with set_root_context(tmp_path / "subdir"):
        dir = RootedDirectory(Namespace(start=0, stop=10, step=1))
        assert dir.path.parent == tmp_path / "subdir"


# -- test default config


class DefaultConfigDir(Directory):
    class Config:
        x: int = 2

    def __init__(self, config) -> None:
        self.x = np.arange(config.x)


class BadImplementation(Directory):
    # Config has no attributes
    class Config:
        pass

    # but has init
    def __init__(self, config) -> None:
        self.x = np.arange(config.x)


def test_default_config(tmp_path):
    set_root_dir(tmp_path)

    # from default config
    dir = DefaultConfigDir()
    assert isinstance(dir, DefaultConfigDir)
    assert (dir.x[()] == np.arange(2)).all()
    assert dir.path.parent == tmp_path
    _first = dir.path.name
    assert "DefaultConfigDir" in _first

    # again
    dir = DefaultConfigDir()
    assert isinstance(dir, DefaultConfigDir)
    assert (dir.x[()] == np.arange(2)).all()
    assert dir.path.parent == tmp_path
    assert _first == dir.path.name

    # from custom config
    dir = DefaultConfigDir(x=3)
    assert isinstance(dir, DefaultConfigDir)
    assert (dir.x[()] == np.arange(3)).all()
    assert dir.path.parent == tmp_path
    assert "DefaultConfigDir" in dir.path.name
    assert _first != dir.path.name

    # with path from default config
    dir = DefaultConfigDir(tmp_path / "test3")
    assert isinstance(dir, DefaultConfigDir)
    assert (dir.x[()] == np.arange(2)).all()
    assert dir.path.parent == tmp_path
    assert "test3" == dir.path.name

    # with path and custom config
    dir = DefaultConfigDir(tmp_path / "test4", dict(x=3))
    assert isinstance(dir, DefaultConfigDir)
    assert (dir.x[()] == np.arange(3)).all()
    assert dir.path.parent == tmp_path
    assert "test4" == dir.path.name

    # with name/ path from custom config but directory exists
    with pytest.raises(FileExistsError):
        dir = DefaultConfigDir(tmp_path / "test3", dict(x=3))

    # bad implementation warning
    with pytest.warns(ImplementationWarning):
        dir = BadImplementation()

    with pytest.warns(ImplementationWarning):
        with pytest.raises(FileNotFoundError):
            dir = BadImplementation(tmp_path / "test8")

    # config has no default attributes but directory has init, with custom config
    with pytest.warns(ImplementationWarning):
        dir = BadImplementation(dict(x=2))
        assert (dir.x[()] == np.arange(2)).all()

    with pytest.warns(ImplementationWarning):
        dir = BadImplementation(tmp_path / "test10", dict(x=2))
        assert (dir.x[()] == np.arange(2)).all()

    # config has no default attributes but directory has init, with custom, wrong config
    with pytest.raises(AttributeError):
        with pytest.warns(ImplementationWarning):
            dir = BadImplementation(dict(y=2))

    with pytest.raises(AttributeError):
        with pytest.warns(ImplementationWarning):
            dir = BadImplementation(tmp_path / "test12", dict(y=2))


# -- test auto docstring


class AutoDocConfigDir(Directory):
    """Dir to test auto docstring based on config."""

    class Config:
        x: int = 2
        y: float = 2.0
        q = Namespace(a=1, b=2)


class SoloConfigDocDir(Directory):
    class Config:
        x: int = 2
        y: float = 2.0
        q = Namespace(a=1, b=2)


class EmptyConfigDocDir(Directory):
    """Dir to test auto docstring based on config."""

    class Config:
        pass


class NoConfigDocDir(Directory):
    """Dir to test auto docstring based on config."""

    pass


def test_auto_doc(tmp_path):
    a = AutoDocConfigDir()

    doc = "Dir to test auto docstring based on config."
    doc += "\n\n"
    doc += "Initialize from config or leave default:\n"
    doc += "{}(dict(\n{}))"
    doc = doc.format(
        "test_directory.AutoDocConfigDir",
        "x: int = 2,\ny: float = 2.0,\nq = Namespace(a=1, b=2),\n",
    )
    assert a.__doc__ == doc

    b = SoloConfigDocDir()
    doc = "Initialize from config or leave default:\n"
    doc += "{}(dict(\n{}))"
    doc = doc.format(
        "test_directory.SoloConfigDocDir",
        "x: int = 2,\ny: float = 2.0,\nq = Namespace(a=1, b=2),\n",
    )
    assert b.__doc__ == doc

    c = EmptyConfigDocDir()
    doc = "Dir to test auto docstring based on config."
    assert c.__doc__ == doc

    d = NoConfigDocDir()
    doc = "Dir to test auto docstring based on config."
    assert d.__doc__ == doc
