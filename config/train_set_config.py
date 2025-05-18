
import random
import numpy as np
import pandas as pd
import torch
import yaml
import env.workflow_scheduling_v3.lib.dataset as dataset


class trainSet_Generate():
    def __init__(self, yaml_path):
        self.trainMatrix = None
        self.trainArrivalTime = None
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            wf_types = 4

            """----train-------"""
            # Create a train_dataset under a specific seed
            if config['env']["generateWay"] == 'fixed':
                train_dataset = np.random.randint(0, wf_types, (1, config['env']['evalNum'], config['env']['wf_num']))
                train_dataset = np.array([list(train_dataset[0]) for _ in range(config['env']['dataGen'] + 1)])
                train_dataset = train_dataset.astype(np.int64)
            else:  # by rotation
                train_dataset = np.random.randint(0, wf_types, (config['env']['dataGen'] + 1, config['env']['evalNum'], config['env']['wf_num']))
                train_dataset = train_dataset.astype(np.int64)
            self.trainMatrix = train_dataset

            # Generate fixed arrival time for training data under a specific seed
            train_shape = train_dataset.shape
            rate = 0.01
            self.trainArrivalTime = np.zeros(train_shape)
            for batch in range(train_shape[0]):
                for row in range(train_shape[1]):
                    time_points = [0]
                    for _ in range(train_shape[2] - 1):
                        time_points.append(time_points[-1] + np.random.exponential(1 / rate))
                    self.trainArrivalTime[batch, row, :] = time_points

            f.close()



