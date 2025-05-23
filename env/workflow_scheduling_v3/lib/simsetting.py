import inspect
import os
import re
import sys

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.dataset import dataset

traffic_density = {"CONSTANT": 1,
                   "LINEAR_INCREASE": [0.1, 0.00025],  # [base_density, increased rate]
                   "LINEAR_DECREASE": [1, -0.00025],
                   "PERIODIC": {0: 0.65, 1: 0.55, 2: 0.35, 3: 0.25, 4: 0.2, 5: 0.16, 6: 0.16, 7: 0.2, 8: 0.4, 9: 0.55,
                                10: 0.65, 11: 0.75, 12: 0.79, 13: 0.79, 14: 0.85, 15: 0.86, 16: 0.85, 17: 0.83, 18: 0.8,
                                19: 0.79, 20: 0.76, 21: 0.76, 22: 0.69, 23: 0.6}}

traffic_type = ["CONSTANT", "LINEAR_INCREASE", "LINEAR_DECREASE", "PERIODIC"]
traffic_dist = ["EVEN", "UNEVEN"]


class Setting(object):
    state_info_sample_period = 50  # state will be recoreded every 50 seconds
    dataformat = "pickle"
    ac_ob_info_required = False
    epsilon = 0  # 0.1  # used in RL for exploration, epsilon greedy
    is_wf_trace_record = False
    save_nn_iteration_frequency = 1  # every 20 timesteps

    def __init__(self, args):
        self.traf_type = args["traffic pattern"]
        self.traf_dist = "EVEN"
        self.seed = args["seed"]
        if "REINFORCE learning rate" in args.keys():
            self.REINFORCE_learn_rate = args["REINFORCE learning rate"]
        if "hist_len" in args.keys():
            self.history_len = args["hist_len"]
        else:
            self.history_len = 2
        self.timeStep = 1800
        self.respTime_update_interval = 0.5
        self.util_update_interval = 0.5
        self.arrival_rate_update_interval = self.timeStep
        self.warmupPeriod = 30  # unit: second
        self.envid = args["envid"]
        self.gamma = args["gamma"]
        self.pkt_trace_sample_freq = 10
        self.VMpayInterval = 60 * 60
        self.dataset = dataset(args["wf_size"])
        self.WorkflowNum = args["wf_num"]
        self._init(self.envid)
        self.dlt = 1.0
        self.mu = [100.0] * 5
        self.beta = 0.1

    def _init(self, num):
        if num == 0:
            latency_matrix = np.array([[0]])
            latency = np.multiply(latency_matrix, 0.5)
            self.candidate = [0]
            self.dcNum = len(self.candidate)
            self.usrNum = latency.shape[0]
            self.candidate.sort()
            self.usr2dc = latency[:, self.candidate]
        else:
            assert (num == 0), "Please set envid to 0!"

        self.wrfNum = len(self.dataset.wset)  
        self.arrival_rate = {}
        for i in range(self.usrNum):
            self.arrival_rate[i] = {}
            for a in range(len(self.dataset.wset)):
                self.arrival_rate[i][a] = self.dataset.request[i]
        self.dueTimeCoef = np.ones((self.usrNum, self.wrfNum)) / max(self.dataset.vmVCPU) * self.gamma
        self.totWrfNum = self.WorkflowNum

    def get_individual_arrival_rate(self, time, usrcenter, app):
        if self.traf_type == "CONSTANT":
            den = traffic_density[self.traf_type]  # default:1
        else:
            if re.match(r"^LINEAR.", self.traf_type):
                den = traffic_density[self.traf_type][0] + traffic_density[self.traf_type][-1] * time
            elif self.traf_type == "PERIODIC":
                hr = int(time / 75) % 24  # we consider two periods in one hour
                den = traffic_density[self.traf_type][hr]
            else:
                print("cannot get the arrival rate!!!!!!!!")
                den = None
        return den * self.arrival_rate[usrcenter][app]
