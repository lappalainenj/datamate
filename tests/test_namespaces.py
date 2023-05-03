import numpy as np
from types import SimpleNamespace

from datamate import Namespace, namespacify

# ------------------------------------------------------------------------------


def test_namespaces():
    ns = Namespace(a=1, b=2)
    assert "a" in dir(ns) and "b" in dir(ns)
    assert ns.a == 1 and ns["a"] == 1
    assert ns.b == 2 and ns["b"] == 2

    ns["a"] = 11
    ns.b = 22
    assert "a" in dir(ns) and "b" in dir(ns)
    assert ns.a == 11 and ns["a"] == 11
    assert ns.b == 22 and ns["b"] == 22

    del ns["a"]
    del ns.b
    assert "a" not in dir(ns) and "b" not in dir(ns)


def test_namespacify():
    obj = {
        "bool": False,
        "int": 1,
        "float": 2.2,
        "str": "three",
        "list": [0, 1, 2],
        "dict": {"a": 0, "b": 1},
        "namespace": SimpleNamespace(c=2, d=3),
        "type": int,
    }
    obj_as_ns = Namespace(
        bool=False,
        int=1,
        float=2.2,
        str="three",
        list=[0, 1, 2],
        dict=Namespace(a=0, b=1),
        namespace=Namespace(c=2, d=3),
        type=int,
    )
    assert namespacify(obj) == obj_as_ns
    assert isinstance(namespacify(obj).dict, Namespace)
    assert isinstance(namespacify(obj).namespace, Namespace)


def test_namespacify_with_nesting():
    obj = {
        "list": [{"a": 0, "b": 1}, SimpleNamespace(c=2, d=3)],
        "dict": {"a": {"a": 0, "b": 1}, "b": SimpleNamespace(c=2, d=3)},
        "namespace": SimpleNamespace(c={"a": 0, "b": 1}, d=SimpleNamespace(c=2, d=3)),
    }
    obj_as_ns = Namespace(
        list=[Namespace(a=0, b=1), Namespace(c=2, d=3)],
        dict=Namespace(a=Namespace(a=0, b=1), b=Namespace(c=2, d=3)),
        namespace=Namespace(c=Namespace(a=0, b=1), d=Namespace(c=2, d=3)),
    )
    assert namespacify(obj) == obj_as_ns
    assert isinstance(namespacify(obj).dict, Namespace)
    assert isinstance(namespacify(obj).dict.a, Namespace)
    assert isinstance(namespacify(obj).dict.b, Namespace)
    assert isinstance(namespacify(obj).namespace, Namespace)
    assert isinstance(namespacify(obj).namespace.c, Namespace)
    assert isinstance(namespacify(obj).namespace.d, Namespace)


def test_repr():
    ns_a = Namespace()
    assert repr(ns_a) == "Namespace()"

    ns_b = Namespace(
        list=[Namespace(a=0, b=1), Namespace(c=2, d=3)],
        dict=Namespace(a=Namespace(a=0, b=1), b=Namespace(c=2, d=3)),
        namespace=Namespace(c=Namespace(a=0, b=1), d=Namespace(c=2, d=3)),
    )
    assert repr(ns_b) == (
        "Namespace(\n"
        "  list = [Namespace(a=0, b=1), Namespace(c=2, d=3)],\n"
        "  dict = Namespace(a=Namespace(a=0, b=1), b=Namespace(c=2, d=3)),\n"
        "  namespace = Namespace(c=Namespace(a=0, b=1), d=Namespace(c=2, d=3))\n"
        ")"
    )

    ns_c = Namespace(
        bool=False,
        int=0,
        float=1.1,
        large_entry=Namespace(
            list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            str="The quick brown fox jumped over the lazy dog.",
        ),
    )
    assert repr(ns_c) == (
        "Namespace(\n"
        "  bool = False,\n"
        "  int = 0,\n"
        "  float = 1.1,\n"
        "  large_entry = Namespace(\n"
        "    list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],\n"
        "    str = 'The quick brown fox jumped over the lazy dog.'\n"
        "  )\n"
        ")"
    )


def test_all():
    x = Namespace(a=Namespace(b=[1, 2, 3]))
    assert x.all()

    x = Namespace(a=Namespace(b=[1, 2, 0]))
    assert not x.all()

    y = x.deepcopy()
    y.a.b = [4, 2, 1]
    assert not y == x
    assert y.all()

    x = Namespace(a=np.arange(3))
    y = Namespace(a=np.arange(1, 5))
    assert not x.all() and y.all()
    assert y != x

    y = Namespace(a=np.arange(3))
    assert y == x
