import time
from pathlib import Path
import numpy as np
import random

from datamate import Directory

cwd = Path(__file__).parent.absolute()

random.seed(0)
np.random.seed(0)


def test_swmr_single_thread(tmp_path):
    directory = Directory(tmp_path / "swmr_single")

    value = np.random.rand(5)
    directory.x = value
    time.sleep(0.1)

    for _ in range(1000):
        operation = random.choice(["read", "read", "write", "extend"])
        if operation == "read":
            reader = directory.x
            read_value = reader[:]
        elif operation == "write":
            write_value = np.random.rand(5)
            directory.x = write_value
            value = write_value
        else:
            extend_value = [np.random.rand()]
            directory.extend("x", extend_value)
            value = np.concatenate([value, np.array(extend_value)])
