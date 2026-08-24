from assembly.assemble_rl import AssembleRL
from utils.utils import get_state_num, get_action_num, is_discrete_action, get_nn_output_num


class Builder:
    def __init__(self, baseconfig, train_Set_setting, test_Set_setting):
        self.config = baseconfig
        self.env = None
        self.policy = None
        self.optim = None

        self.train_Set_setting = train_Set_setting
        self.test_Set_setting = test_Set_setting

    def build(self):
        if self.config.config['yaml-config']["env"]["gamma_train"] is not None:
            self.config.config['yaml-config']["env"]["gamma"] = self.config.config['yaml-config']["env"]["gamma_train"]
        env = build_env(self.config.config, self.train_Set_setting, self.test_Set_setting)

        if self.config.config['yaml-config']["env"]["gamma_test"] is not None:
            self.config.config['yaml-config']["env"]["gamma"] = self.config.config['yaml-config']["env"]["gamma_test"]
        env_test = build_env(self.config.config, self.train_Set_setting, self.test_Set_setting)

        self.config.config['yaml-config']["policy"]["discrete_action"] = is_discrete_action(env)
        self.config.config['yaml-config']["policy"]["state_num"] = get_state_num(env)
        self.config.config['yaml-config']["policy"]["action_num"] = get_nn_output_num(env)

        policy = build_policy(self.config.config['yaml-config']["policy"])
        optim = build_optim(self.config.config['yaml-config']["optim"])

        return AssembleRL(self.config, env, env_test, policy, optim)

def build_env(config, train_Set_setting, test_Set_setting):
    env_name = config['yaml-config']["env"]["name"]
    config['yaml-config']['env']['evalNum'] = config['runtime-config']['eval_ep_num']
    if env_name == "WorkflowScheduling-v3":
        from env.workflow_scheduling_v3.simulator_wf import WFEnv
        return WFEnv(env_name, config['yaml-config']["env"], train_Set_setting, test_Set_setting)
    else:
        raise AssertionError(f"{env_name} doesn't support, please specify supported a env in yaml.")

def build_policy(config):
    model_name = config["name"]
    if model_name == "model_workflow":
        from policy.wf_model_02 import WFPolicy
        return WFPolicy(config)
    else:
        raise AssertionError(f"{model_name} doesn't support, please specify supported a model in yaml.")

def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es_openai":
        from optim.es.es_openai_03 import ESOpenAI
        return ESOpenAI(config)
    else:
        raise AssertionError(f"{optim_name} doesn't support, please specify supported a optim in yaml.")
