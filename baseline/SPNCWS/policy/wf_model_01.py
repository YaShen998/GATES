"""
Algorithm: SPNCWS 
Paper: Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning. ICSOC 2024.
Authors: Ya Shen, Gang Chen, Hui Ma, and Mengjie Zhang
Note: You need to adapt the algorithm to suit your specific problems and configurations to ensure fairness.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy


class SelfAttentionEncoder(nn.Module):
    def __init__(self, task_fea_size, vm_fea_size, output_size, d_model, num_heads, num_en_layers, d_ff, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()

        self.task_embedding = nn.Sequential(nn.Linear(task_fea_size, d_model))
        self.task_feature_enhance = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.ReLU(), nn.Linear(2*d_model, d_model))

        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_en_layers)

        self.priority = nn.Sequential(nn.Linear(2 * d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, output_size))

    def forward(self, task_info, vm_info):
        task_embedded = self.task_embedding(task_info)
        task_feature_enhance = self.task_feature_enhance(task_embedded)

        vm_embedded = self.vm_embedding(vm_info)
        vm_embedded = vm_embedded.permute(1, 0, 2)
        global_info = self.transformer_encoder(vm_embedded)
        global_info = global_info.permute(1, 0, 2)

        concatenation_features = torch.cat((global_info, task_feature_enhance), dim=-1)
        priority = self.priority(concatenation_features)
        return priority


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

        self.model = SelfAttentionEncoder(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16, num_heads=2, num_en_layers=2, d_ff=64, dropout=0.1)

    def forward(self, ob, dag, node_id, VM_features_matrix, removeVM=None):
        with (torch.no_grad()):
            x = torch.from_numpy(ob).float()
            task_info = x[:, 0:-4].unsqueeze(1)
            vm_info = x[:, -4::].unsqueeze(1)
            x = self.model(task_info, vm_info)
            x = x.permute(1, 0, 2)

            if removeVM is not None:
                x[:, removeVM, :] = float("-inf")

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.relu(x.squeeze())

            x = x.detach().cpu().numpy()
            return x.item(0)

    def forward_reinforce(self, x, removeVM=None):
        x = torch.from_numpy(x).float()
        task_info = x[0, 0:-4].view(4, 1).unsqueeze(1)
        vm_info = x[:, -4::].unsqueeze(1)
        x = self.model(task_info, vm_info)
        x = x.permute(1, 0, 2)

        if removeVM is not None:
            x[:, removeVM, :] = float("-inf")

        if self.discrete_action:
            probabilities = F.softmax(x.squeeze(), dim=0)
            m = torch.distributions.Categorical(probabilities)
            action = m.sample()
            log_prob = m.log_prob(action)
            return action.item(), log_prob
        else:
            pass

    def forward_reinforce2(self, x, all_len_state, all_actions):
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        task_info = x[0, 0:-4].view(4, 1).unsqueeze(1)
        vm_info = x[:, -4::].unsqueeze(1)
        x = self.model(task_info, vm_info)
        x = x.permute(1, 0, 2)

        flat_list_all_len_state = [item for sublist in all_len_state for item in sublist]
        sparse_matrix = create_sparse_matrix(flat_list_all_len_state, x.shape[0])
        x_exp = torch.exp(x)
        result = torch.sparse.mm(sparse_matrix, x_exp)
        result_2 = x_exp / result

        flat_list_all_actions = [item for sublist in all_actions for item in sublist]
        flat_list_all_actions_tensor = torch.tensor(flat_list_all_actions, dtype=torch.int64)
        flat_list_all_len_state_tensor = torch.tensor(flat_list_all_len_state, dtype=torch.int64)

        rows = torch.arange(len(flat_list_all_len_state_tensor), dtype=torch.int64)
        cumulative_lengths = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), flat_list_all_len_state_tensor[:-1]]), dim=0)
        cols = cumulative_lengths + flat_list_all_actions_tensor

        indices = torch.stack([rows, cols])
        values = torch.ones(len(flat_list_all_len_state_tensor))
        selection_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([len(flat_list_all_len_state_tensor), result_2.shape[0]]))
        result = torch.sparse.mm(selection_matrix, result_2)

        extracted_elements = result.flatten()
        log_prob_re = torch.log(extracted_elements)
        return log_prob_re

    def forward_(self, x):
        x = torch.from_numpy(x).float()
        task_info = x[0, 0:-4].view(4, 1).unsqueeze(1)
        vm_info = x[:, -4::].unsqueeze(1)
        x = self.model(task_info, vm_info)
        x = x.permute(1, 0, 2)

        x = torch.reshape(x, (1, -1))
        return x

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

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

    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float)
    return torch.sparse.FloatTensor(indices, values, torch.Size([rows, length]))
