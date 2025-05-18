
import random
import torch
from env.workflow_scheduling_v3.lib.cloud_env_maxPktNum import cloud_simulator
import numpy as np
import env.workflow_scheduling_v3.lib.dataset as dataset


class WFEnv(cloud_simulator):
    def __init__(self, name, args, train_Set_setting, test_Set_setting):
        # load the fixed train_dataset and test_dataset
        self.train_Set_setting = train_Set_setting
        self.test_Set_setting = test_Set_setting

        trainMatrix = self.train_Set_setting.trainMatrix
        trainArrivalTime = self.train_Set_setting.trainArrivalTime
        testMatrix = self.test_Set_setting.testMatrix
        testArrivalTime = self.test_Set_setting.testArrivalTime

        # Setup
        config = {"traffic pattern": args['traffic_pattern'], "seed": args['seed'], "gamma": args['gamma'],
                  "envid": 0, "wf_size": args["wf_size"], "wf_num": args["wf_num"],
                  "trainSet": trainMatrix, 'trainArrivalTime': trainArrivalTime,
                  'testSet': testMatrix, 'testArrivalTime': testArrivalTime}

        super(WFEnv, self).__init__(config)
        self.name = name

    def reset(self, seed=None, ep_num=None, train_or_test=None):
        super(WFEnv, self).reset(seed, ep_num, train_or_test)
        # the seed of env should be fixed across different training seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.step_curr = self.numTimestep
        state_dict_list = {}
        state_dict = {}
        s, dag, node_id = self.state_info_construct()
        state_dict["state"] = np.array(s)
        state_dict["DAG"] = dag
        state_dict["Node_id"] = node_id
        state_dict_list["0"] = state_dict
        return state_dict_list

    def step(self, action):
        r, usr_respTime, usr_received_appNum, usr_sent_pktNum, d = super(WFEnv, self).step(action["0"])
        state_dict_list = {}
        state_dict = {}
        s, dag, node_id = self.state_info_construct()
        info = [usr_respTime, usr_received_appNum, usr_sent_pktNum]

        state_dict["state"] = np.array(s)
        state_dict["DAG"] = dag
        state_dict["Node_id"] = node_id
        state_dict["reward"] = r
        state_dict["done"] = d
        state_dict["info"] = info
        state_dict_list["0"] = state_dict

        if hasattr(self, "VMtobeRemove"):
            # get the vm index
            state_dict_list["removeVM"] = self.vm_queues_id.index(self.VMtobeRemove) if self.VMtobeRemove in self.vm_queues_id else None

        return state_dict_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def close(self):
        super(WFEnv, self).close()

