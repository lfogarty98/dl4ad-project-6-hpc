#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, hidden_size, num_lstm_layers):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S model.hidden_size={hidden_size} -S model.num_lstm_layers={num_lstm_layers}"
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    hidden_size_list = [500, 750, 1000]
    num_lstm_layers_list = [3, 5, 7]
    # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    for hidden_size, num_lstm_layers in itertools.product(hidden_size_list, num_lstm_layers_list):
        submit_batch_job(arguments, hidden_size, num_lstm_layers)
