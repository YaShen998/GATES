import multiprocessing as mp
import os
import shutil
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pickle as pickle
import yaml

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd

from utils.policy_dict import agent_policy
import builder


class AssembleRL(BaseAssembleRL):
    def __init__(self, config, env, env_test, policy, optim):
        super(AssembleRL, self).__init__()
        self.config = config
        self.env = env
        self.env_test = env_test
        self.policy = policy
        self.optim = optim

        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:
            self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']['optim']['generation_num']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.eval_ep_num = self.config.config['runtime-config']['eval_ep_num']
        self.valid_ep_num = self.config.config['yaml-config']['env']['validNum']

        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None

        self.train_or_test = None

    def train(self):
        if self.log:
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/{self.env.name}/{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)

            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        # Initialize the policy population.
        population = self.optim.init_population(self.policy, self.env)

        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        # Roll out and update each generation.
        for g in range(self.generation_num):
            start_time = time.time()

            self.train_or_test = 'train'
            env = self.env
            arguments = [(indi, env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.config, self.train_or_test) for indi in population]

            if self.config.config['yaml-config']['env']['system'] == "Windows":
                start_time_rollout = time.time()
                if self.processor_num > 1:
                    p = mp.get_context('spawn').Pool(self.processor_num)
                    results = p.map(worker_func, arguments)
                    p.close()
                    p.join()
                else:
                    results = [worker_func(arg) for arg in arguments]
            else:
                start_time_rollout = time.time()
                if self.processor_num > 1:
                    p = mp.get_context('fork').Pool(self.processor_num)
                    results = p.map(worker_func, arguments)
                    p.close()
                    p.join()
                else:
                    results = [worker_func(arg) for arg in arguments]

            end_time_rollout = time.time() - start_time_rollout

            start_time_eval = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])

            population, sigma_curr, best_reward_per_g = self.optim.next_population(self, results_df, g)
            end_time_eval = time.time() - start_time_eval

            end_time_generation = time.time() - start_time

            maximization = self.config.config['yaml-config']['optim']['maximization']
            if (maximization and best_reward_per_g > best_reward_so_far) or (not maximization and best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            print(
                f"\nepisode: {g}, gamma:{env.set.gamma}, [current_policy_population:], best reward so far: {best_reward_so_far:.4f}, "
                f"best reward of the current generation: {best_reward_per_g:.4f}, sigma: {sigma_curr:.3f}, "
                f"time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}", flush=True
            )

            training_reward, training_VM_cost, training_SLA_penalty = (results_df[col].tolist()[0] for col in ['rewards', 'VM_cost', 'SLA_penalty'])
            print(
                f"episode: {g}, gamma:{env.set.gamma}, [the_basic_policy:], current training reward: {training_reward:.4f}, "
                f"current training VM_cost: {training_VM_cost:.4f}, current training SLA_penalty: {training_SLA_penalty:.4f}", flush=True
            )

            # Update observation normalization.
            if self.running_mstd:
                hist_obs = []
                hist_obs = np.concatenate(results_df['hist_obs'], axis=0)

                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            # Save training results.
            if self.log:
                if self.running_mstd:
                    results_df = results_df.drop(['hist_obs'], axis=1)

                results_df = results_df.loc[results_df['policy_id'] == -1]

                dir_train = self.save_mode_dir + "/train_performance"
                if not os.path.exists(dir_train):
                    os.makedirs(dir_train)
                results_df.to_csv(dir_train + "/training_record.csv", index=False, header=False, mode='a')

                elite = self.optim.get_elite_model()
                if (g + 1) % self.save_model_freq == 0 or g == 0:
                    if g == 0:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{g}.pt"
                    else:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g + 1)}.pt"
                    torch.save(elite.state_dict(), save_pth)
                    if self.running_mstd:
                        if g == 0:
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}.pickle"
                        else:
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g + 1)}.pickle"
                        f = open(save_pth, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

            # Evaluate saved checkpoints.
            if ((g+1) % self.save_model_freq) == 0 or g == 0:
                from utils.policy_dict import agent_policy
                indi_test = []
                agent_ids_test = self.env.get_agent_ids()
                model_test = self.optim.get_elite_model()
                indi_test.append(agent_policy(agent_ids_test, model_test))
                self.train_or_test = 'test'
                env = self.env_test
                arguments = [(indi, env, self.optim, self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std,
                              self.processor_num, 0, self.config, self.train_or_test) for indi in indi_test]

                start_time_test = time.time()
                results = [worker_func(arg) for arg in arguments]
                end_time_test = time.time() - start_time_test
                results_df = pd.DataFrame(results)

                testing_reward = results_df['rewards'].tolist()[0]
                VM_cost = results_df["VM_cost"].tolist()[0]
                SLA_penalty = results_df["SLA_penalty"].tolist()[0]
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(
                    f"episode: {g}, gamma:{env.set.gamma}, [<<<<----testing---->>>>], current testing reward: {testing_reward:.4f}, "
                    f"current testing VM_cost: {VM_cost:.4f}, current testing SLA_penalty: {SLA_penalty:.4f}, "
                    f"current testing_time: {end_time_test:.2f}",
                    flush=True
                )

                if self.log:
                    results_df = results_df.drop(['hist_obs'], axis=1)
                    dir_test = self.save_mode_dir + "/test_performance"
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(dir_test + "/testing_record_in_training.csv", index=False, header=False, mode='a')

    def eval(self):
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))

        env = self.env_test

        indi = agent_policy(env.get_agent_ids(), self.policy)

        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()

        g = 0
        self.train_or_test = 'test'

        arguments = [(indi, env, self.optim, self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std, self.processor_num, g, self.config, self.train_or_test)]

        start_time_test = time.time()

        results = [worker_func(arg) for arg in arguments]

        end_time_test = time.time() - start_time_test

        results_df = pd.DataFrame(results)

        testing_reward = results_df['rewards'].tolist()[0]
        VM_cost = results_df["VM_cost"].tolist()[0]
        SLA_penalty = results_df["SLA_penalty"].tolist()[0]
        print(
            f"gamma:{env.set.gamma}, current testing reward: {testing_reward:.4f}, current VM cost: {VM_cost:.4f}, "
            f"current SLA penalty: {SLA_penalty:.4f}, testing_time: {end_time_test:.2f}\n", flush=True
        )

        if self.log:
            results_df = results_df.drop(['hist_obs'], axis=1)
            if self.config.config['runtime-config']['final'] == True:
                dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance_final"

                test_size = self.config.config['yaml-config']['env']['wf_size']
                gamma_size = self.config.config['yaml-config']['env']['gamma_test']
                model_num = self.config.config['yaml-config']['env']['model_num']
                if not os.path.exists(dir_test):
                    os.makedirs(dir_test)
                results_df.to_csv(
                    dir_test + "/testing_record_"+str(gamma_size)+"_"+str(test_size)+"_"+str(model_num)+".csv",
                    index=False, header=False, mode='a')
            else:
                dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"

                test_size = self.config.config['yaml-config']['env']['wf_size']
                gamma_size = self.config.config['yaml-config']['env']['gamma_test']
                if not os.path.exists(dir_test):
                    os.makedirs(dir_test)
                results_df.to_csv(dir_test + "/testing_record_"+str(gamma_size)+"_"+str(test_size)+".csv", index=False, header=False, mode='a')

def worker_func(arguments):
    indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config, train_or_test = arguments

    env = builder.build_env(config.config, env.train_Set_setting, env.test_Set_setting)

    hist_rewards = {}
    hist_obs = {}
    hist_actions = {}
    obs = None
    total_reward = 0

    total_VM_execHour = 0
    total_VM_totHour = 0
    total_VM_cost = 0
    total_SLA_penalty = 0
    total_missDeadlineNum = 0

    # Collect rollout metrics.
    for ep_num in range(eval_ep_num):
        states = env.reset(g, ep_num, train_or_test)

        rewards_per_eval = []
        obs_per_eval = []
        actions_per_eval = []
        done = False

        for agent_id, model in indi.items():
            model.reset()
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states[agent_id]["state"]
                dag = states[agent_id]["DAG"]
                node_id = states[agent_id]["Node_id"]
                VM_configuration = states[agent_id]["VM_configuration"]

                if s.ndim < 2:
                    s = s[np.newaxis, :]

                if ob_rms_mean is not None:
                    s = (s - ob_rms_mean) / ob_rms_std

                if "removeVM" in states:
                    actions[agent_id] = model(s, dag, node_id, VM_configuration, removeVM=states["removeVM"])
                else:
                    actions[agent_id] = model(s, dag, node_id, VM_configuration)

                states, r, done, _ = env.step(actions)

                rewards_per_eval.append(r)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                total_reward += r

                if obs is None:
                    obs = states["0"]["state"]
                else:
                    obs = np.append(obs, states["0"]["state"], axis=0)

        hist_rewards[ep_num] = rewards_per_eval
        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval

        total_VM_execHour += env.episode_info["VM_execHour"]
        total_VM_totHour += env.episode_info["VM_totHour"]
        total_VM_cost += env.episode_info["VM_cost"]
        total_SLA_penalty += env.episode_info["SLA_penalty"]
        total_missDeadlineNum += env.episode_info["missDeadlineNum"]

    rewards_mean = total_reward / eval_ep_num

    VM_execHour_mean = total_VM_execHour / eval_ep_num
    VM_totHour_mean = total_VM_totHour / eval_ep_num
    VM_cost_mean = total_VM_cost / eval_ep_num
    SLA_penalty_mean = total_SLA_penalty / eval_ep_num
    missDeadlineNum_mean = total_missDeadlineNum / eval_ep_num

    if env.name in ["WorkflowScheduling-v0", "WorkflowScheduling-v2", "WorkflowScheduling-v3"] and optim.name == "es_openai":
        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": VM_execHour_mean,
                    "VM_totHour": VM_totHour_mean,
                    "VM_cost": VM_cost_mean,
                    "SLA_penalty": SLA_penalty_mean,
                    "missDeadlineNum": missDeadlineNum_mean}
        else:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan}

def discount_rewards(rewards):
    gamma = 0.99
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards
