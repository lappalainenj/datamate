import time
import sys
from pathlib import Path
import subprocess
import random
import pytest
import numpy as np
import multiprocessing

from datamate import Directory
from datamate.directory import ConfigWarning
from test_directory import assert_directory_equals

cwd = Path(__file__).parent.absolute()

random.seed(0)
np.random.seed(0)


def run_swmr_process(args):
    # all timings arbitrary
    path, mode = args
    directory = Directory(path)
    if mode == "read":
        time.sleep(random.random() * 2)
        handle = directory.x
        read_value = handle[()]
        time.sleep(random.random() * 2) # artificially make read file handle persist
    elif mode == "write":
        start_time = time.time()
        while time.time() - start_time < 30: # run write for 30 s. ideally would run until all reads done, but too lazy to implement
            time.sleep(random.random() / 5 + 0.05)
            if random.random() < 0.5: # write or extend
                directory.x = np.random.rand(5)
            else:
                directory.extend('x', [random.random()])


def test_swmr_single_thread(tmp_path):
    directory = Directory(tmp_path / "swmr_single")

    value = np.random.rand(5)
    directory.x = value
    time.sleep(0.1)

    for _ in range(10000):
        operation = random.choice(["read", "read", "write", "write"])
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


def test_swmr_multi_thread(tmp_path):
    directory = Directory(tmp_path / "swmr_single")

    value = np.random.rand(5)
    directory.x = value
    time.sleep(0.1)
    
    with multiprocessing.Pool(100) as p:
        p.map(
            run_swmr_process, 
            [
                ((tmp_path / "swmr_single", "write" if i == 0 else "read")) # one write job, many reads
                for i in range(100)
            ],
        )