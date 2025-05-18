
"""
Algorithm: ES_RL
Paper: Cost-aware dynamic multi-workflow scheduling in cloud data center using evolutionary reinforcement learning. ICSOC 2022.
Authors: Victoria Huang, Chen Wang, Hui Ma, Gang Chen, and Kameron Christopher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy


class WFPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(WFPolicy, self).__init__()
        self.policy_id = policy_id
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True

        self.fc1 = nn.Linear(self.state_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_num)

    def forward(self, ob, dag, node_id, removeVM=None):
        with torch.no_grad():
            x = torch.from_numpy(ob).float()
            x = x.unsqueeze(0)  # Todo: check x dim as its condition
            x = torch.tanh(self.fc1(x))  # set to tanh in Victoria's paper
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)

            if removeVM is not None:
                x[:, removeVM, :] = float("-inf")

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)

            else:
                x = torch.relu(x.squeeze())

            x = x.detach().cpu().numpy()

            return x.item(0)

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)  # if relu used, bias is set to 0.01

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1

def create_sparse_matrix(ranges, length):
    rows = sum(ranges)
    indices = []
    values = []
    start_index = 0
    for range_length in ranges:
        for i in range(range_length):
            for j in range(start_index, start_index + range_length):
                indices.append([start_index + i, j])
                values.append(1)
        start_index += range_length
    # indices = torch.LongTensor(indices).t()
    # values = torch.FloatTensor(values)
    # return torch.sparse.FloatTensor(indices, values, torch.Size([rows, length]))
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float)
    # return torch.sparse.FloatTensor(indices, values, torch.Size([rows, length]))
    return torch.sparse_coo_tensor(indices, values, torch.Size([rows, length]))