#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, test_split, batch_size, hidden_size):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S preprocess.test_split={test_split} -S train.batch_size={batch_size} -S model.hidden_size={hidden_size}"
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    test_split_list = [0.2, 0.3]
    batch_size_list = [64, 128]
    hidden_size_list = [50, 100, 150]
    
    # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    for test_split, batch_size in itertools.product(test_split_list, batch_size_list, hidden_size_list):
        submit_batch_job(arguments, test_split, batch_size, hidden_size_list)
