#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, lambda_reg, max_voices):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S train.lambda_reg={lambda_reg} -S train.max_voices={max_voices}"
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    lambda_reg_list = [0.1, 0.3]
    max_voices_list = [1, 2]
    # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    for lambda_reg, max_voices in itertools.product(lambda_reg_list, max_voices_list):
        submit_batch_job(arguments, lambda_reg, max_voices)
