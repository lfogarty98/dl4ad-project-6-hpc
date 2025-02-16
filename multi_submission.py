#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, max_voices, batch_size):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S train.max_voices={max_voices} -S train.batch_size={batch_size}"
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    # hidden_size_list = [400, 600]
    batch_size_list = [2048, 4096]
    max_voices_list = [3, 4, 5]
    # learning_rate_list = [0.001, 0.005]
    # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    for max_voices, batch_size in itertools.product(max_voices_list, batch_size_list):
        submit_batch_job(arguments, max_voices, batch_size)
