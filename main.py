import random
import numpy as np
import torch
from config.base_config import BaseConfig
from builder import Builder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(test_Set_setting):
    baseconfig = BaseConfig()

    # Set global running seed
    set_seed(baseconfig.config["yaml-config"]['env']['seed'])

    # Create a training set under current seed
    from config.train_set_config import trainSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    train_Set_setting = trainSet_Generate(yaml_path)

    # Start assembling RL and training process
    Builder(baseconfig, train_Set_setting, test_Set_setting).build().train()


if __name__ == "__main__":
    # Create the fixed testing set
    from config.test_set_config import testSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    test_Set_setting = testSet_Generate(yaml_path)

    main(test_Set_setting)
