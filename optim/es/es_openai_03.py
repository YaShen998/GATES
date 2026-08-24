"""
    Update policy parameters with ES using PyTorch's built-in Adam
"""

from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from optim.base_optim import BaseOptim
from utils.optimizers import Adam
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params
from collections import deque


class ESOpenAI(BaseOptim):
    def __init__(self, config):
        super(ESOpenAI, self).__init__()
        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.reinforce_learning_rate = config["reinforce_learning_rate"]
        self.population_size = config["population_size"]
        self.reward_shaping = config['reward_shaping']
        self.reward_norm = config['reward_norm']

        self.epsilons = []

        self.agent_ids = None
        self.mu_model = None
        self.mu_model_flatten_params = None
        self.optimizer = None

    def init_population(self, policy: torch.nn.Module, env):
        self.agent_ids = env.get_agent_ids()
        policy.norm_init()
        self.mu_model = policy
        self.mu_model_flatten_params = torch.tensor(get_flatten_params(self.mu_model)['params'], requires_grad=True, dtype=torch.float64)
        self.optimizer = torch.optim.Adam([self.mu_model_flatten_params], lr=self.learning_rate)

        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)
        return perturbations

    def init_perturbations(self, agent_ids: list, mu_model: torch.nn.Module, sigma, pop_size):
        perturbations = []
        self.epsilons = []

        perturbations.append(agent_policy(agent_ids, mu_model))

        zero_eps = deepcopy(mu_model)
        zero_eps.zero_init()
        zero_eps_param_lst = get_flatten_params(zero_eps)
        self.epsilons.append(zero_eps_param_lst['params'])

        for _num in range(pop_size):
            perturbed_policy = deepcopy(mu_model)
            perturbed_policy.set_policy_id(_num)

            perturbed_policy_param_lst = get_flatten_params(perturbed_policy)
            epsilon = np.random.normal(size=perturbed_policy_param_lst['params'].shape)
            perturbed_policy_param_updated = perturbed_policy_param_lst['params'] + epsilon * sigma

            set_flatten_params(perturbed_policy_param_updated, perturbed_policy_param_lst['lengths'], perturbed_policy)

            perturbations.append(agent_policy(agent_ids, perturbed_policy))
            self.epsilons.append(epsilon)

        return perturbations

    def next_population(self, assemble, results, g):
        rewards = results['rewards'].tolist()
        best_reward_per_g = max(rewards)
        rewards = np.array(rewards)

        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)

        if self.reward_norm:
            r_std = rewards.std()
            rewards = (rewards - rewards.mean()) / r_std

        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)
        update_factor *= -1.0

        grad_param_list = np.sum(np.array(self.epsilons) * rewards.reshape(rewards.shape[0], 1), axis=0)
        grad_param_list *= update_factor

        self.mu_model_flatten_params.grad = torch.tensor(grad_param_list, dtype=torch.float64)
        self.optimizer.step()
        flatten_params = self.mu_model_flatten_params.clone()
        set_flatten_params(flatten_params.detach().numpy(), get_flatten_params(self.mu_model)['lengths'], self.mu_model)

        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, best_reward_per_g

    def get_elite_model(self):
        return self.mu_model

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
