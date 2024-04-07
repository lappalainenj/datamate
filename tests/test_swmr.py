import time
from pathlib import Path
import subprocess
import pytest
import numpy as np
import random

from datamate import Directory
from datamate.directory import ConfigWarning
from test_directory import assert_directory_equals

cwd = Path(__file__).parent.absolute()

random.seed(0)
np.random.seed(0)


def run_swmr_process(tmp_path, mode, args=[]):
    command = ["python", f"{cwd}/swmr.py", str(tmp_path.absolute()), mode] + args
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    return proc


def test_swmr_single_thread(tmp_path):
    directory = Directory(tmp_path / "swmr_dir")

    value = np.random.rand(5)
    directory.x = value
    time.sleep(0.1)

    for _ in range(10000):
        operation = random.choice(["read", "read", "write", "extend"])
        if operation == "read":
            reader = directory.x
            read_value = reader[:]
            # assert np.all(value == read_value)
        elif operation == "write":
            write_value = np.random.rand(5)
            directory.x = write_value
            value = write_value
        else:
            extend_value = [np.random.rand()]
            directory.extend('x', extend_value)
            value = np.concatenate([value, np.array(extend_value)])


def test_swmr_multithread(tmp_path):
    n_readers = 3

    # start the writer process
    writer = run_swmr_process(tmp_path / "swmr_dir", "write", [str(n_readers)])
    # wait for the writer to start
    time.sleep(1.0)

    readers = []
    # start multiple reader readers
    for i in range(n_readers):
        reader = run_swmr_process(
            tmp_path / "swmr_dir", "read", [str(n_readers), str(i)]
        )
        # wait for the reader to start
        time.sleep(1.0)
        readers.append(reader)

    writer.wait(20)
    output, errors = writer.communicate()
    if writer.returncode == 0:
        pass
    else:
        raise AssertionError(f"Writer process failed: {errors.decode('utf-8')}")

    # Wait for readers to finish and capture their output
    for reader in readers:
        reader.wait(1)
        output, errors = reader.communicate()
        # assert "ConfigWarning" in errors.decode("utf-8")
        if reader.returncode == 0:
            pass
        else:
            raise AssertionError(f"Reader process failed: {errors.decode('utf-8')}")

    with pytest.warns(ConfigWarning):
        dir = Directory(tmp_path / "swmr_dir")

    readouts = {f"x{i}": dir[f"x{i}"][:] for i in range(n_readers)}

    def strictly_increasing(L):
        return all(x < y for x, y in zip(L, L[1:]))

    assert strictly_increasing([max(v) for v in readouts.values()])

    assert_directory_equals(
        dir,
        dict(
            __path__=tmp_path / "swmr_dir",
            __conf__=dict(type="Writer", N=10000, sleep=0.01),
            __exists__=True,
            __meta__={
                "config": {"type": "Writer", "N": 10000, "sleep": 0.01},
                "status": "done",
            },
            x=dir.x[:],
            **readouts,
        ),
    )
