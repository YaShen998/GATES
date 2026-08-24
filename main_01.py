"""
    Main Script for SSH
"""

import argparse
import random
import numpy as np
import torch
from config.base_config import BaseConfig
from builder import Builder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(run, test_setting):
    baseconfig = BaseConfig()

    # Set the run-specific seed.
    set_seed(run*100+1)
    print(f"seed:{run*100+1}")

    from config.train_set_config import trainSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    train_Set_setting = trainSet_Generate(yaml_path)

    # Build and run the training pipeline.
    Builder(baseconfig, train_Set_setting, test_setting).build().train()

if __name__ == "__main__":
    from config.test_set_config import testSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    test_Set_setting = testSet_Generate(yaml_path)

    NeSi_parser = argparse.ArgumentParser(description='settings func', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    NeSi_parser.add_argument('--run', '-r', type=int, required=True, help='the run number')
    NeSi_args = NeSi_parser.parse_args()
    print(f"run:{NeSi_args.run}")

    main(NeSi_args.run, test_Set_setting)
