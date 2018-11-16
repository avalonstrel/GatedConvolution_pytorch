import numpy as np
import sys
import os

model_dir = "model_logs"
for log_dir in os.listdir(model_dir):
    log_files = list(os.listdir(os.path.join(model_dir, log_dir)))
    if len(log_files) <= 1:
        os.system("rm -r {}".format(os.path.join(model_dir, log_dir)))
    else:
        print(log_dir, log_files)
