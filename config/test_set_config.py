
"""
    Create a fixed matrix to generate a fixed test_set and fix the arrival time of each workflow
"""
import random
import numpy as np
import pandas as pd
import torch
import yaml
import env.workflow_scheduling_v3.lib.dataset as dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class testSet_Generate():
    def __init__(self, yaml_path):
        self.testMatrix = None
        self.testArrivalTime = None

        set_seed(42)  # fixed the testing set for reproducibility

        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            wf_types = 4
            """----test_set------"""
            # Generate test matrix
            wf_num_testing = config['env']['wf_num_test']
            validNum = config['env']['validNum']
            self.testMatrix = np.random.randint(0, wf_types, (1, validNum, wf_num_testing))

            # Generate Poisson time matrix for each workflow for reproducibility
            test_shape = self.testMatrix.shape
            rate = 0.01
            self.testArrivalTime = np.zeros(test_shape)
            for batch in range(test_shape[0]):
                for row in range(test_shape[1]):
                    time_points = [0]
                    for _ in range(test_shape[2] - 1):
                        time_points.append(time_points[-1] + np.random.exponential(1 / rate))
                    self.testArrivalTime[batch, row, :] = time_points

            f.close()



