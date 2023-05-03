import sys
import time
import pytest
from datamate import Directory
from datamate.directory import ConfigWarning

if __name__ == "__main__":
    args = sys.argv[1:]

    tmp_path = args[0]
    mode = args[1]
    n_readouts = int(args[2])

    def _check_readouts(dir):
        return all([f"x{i}" in dir for i in range(n_readouts)])

    class Writer(Directory):
        def __init__(self, N=10, sleep=0.01):
            for i in range(N):
                self.extend("x", [i])
                time.sleep(sleep)
                # break writing if all readouts have been written
                if _check_readouts(self):
                    break

    if mode == "write":
        # launch writing process that runs until all readouts have been written
        dir = Writer(tmp_path, N=10000, sleep=0.01)
    elif mode == "read":
        # read data and save to readout
        dir = Directory(tmp_path)
        assert dir.status == "running"
        readout = dir.x[:]
        dir[f"x{args[3]}"] = readout
