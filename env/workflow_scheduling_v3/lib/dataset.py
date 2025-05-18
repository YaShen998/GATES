
import inspect
import os
import sys
import numpy as np
from env.workflow_scheduling_v3.lib.buildDAGfromXML import buildGraph
from env.workflow_scheduling_v3.lib.get_DAGlongestPath import get_longestPath_nodeWeighted

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

dataset_30 = ['CyberShake_30', 'Montage_25', 'Inspiral_30', 'Sipht_30']  # test instance 1
dataset_50 = ['CyberShake_50', 'Montage_50', 'Inspiral_50', 'Sipht_60']  # test instance 2
dataset_100 = ['CyberShake_100', 'Montage_100', 'Inspiral_100', 'Sipht_100']  # test instance 3

dataset_dict = {'S': dataset_30, 'M': dataset_50, 'L': dataset_100}


class dataset:
    def __init__(self, arg):
        if arg not in dataset_dict:
            raise NotImplementedError
        self.wset = []
        self.wsetTotProcessTime = []
        for i, j in zip(['CyberShake', 'Montage', 'Inspiral', 'Sipht'], dataset_dict[arg]):
            dag, wsetProcessTime = buildGraph(f'{i}', parentdir + f'/workflow_scheduling_v3/dax/{j}.xml')
            self.wset.append(dag)
            self.wsetTotProcessTime.append(wsetProcessTime)

        self.wsetSlowestT = []
        for app in self.wset:
            self.wsetSlowestT.append(get_longestPath_nodeWeighted(app))

        self.wsetBeta = []
        for app in self.wset:
            self.wsetBeta.append(2)

        self.vmVCPU = [2, 4, 8, 16, 32, 48]  # EC2 m5

        self.request = np.array([1]) * 0.01  # Poisson's distribution: have already generated in config directory

        self.datacenter = [(0, 'East, USA', 0.096)]

        self.vmPrice = {2: 0.096, 4: 0.192, 8: 0.384, 16: 0.768, 32: 1.536, 48: 2.304}
