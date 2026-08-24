import os
import time

run = 35

# Submit training jobs.
for i in range(run):
    os.system(f"sbatch myjob.sl {i+1}")
    time.sleep(0.1)
