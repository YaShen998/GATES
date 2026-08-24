"""
    This script is used to evaluate the model with different gamma and sizes on each episode model.
"""

import argparse
import numpy as np
import os
import random
import torch
from builder import Builder
from config.eval_config_01 import EvalConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(NeSi_args, test_Set_setting):
    print(f"gamma:{NeSi_args.gamma}")
    print(f"wf_size:{NeSi_args.wf_size}\n")

    model_count = 0
    model_num = int(NeSi_args.model_num)
    for fr in np.arange(0, 2020, 20, dtype=int):
        model = f'{NeSi_args.log_path}/saved_models/ep_{fr}.pt'
        model_count += 1
        if not os.path.exists(model):
            break
        eval_config = EvalConfig(model_count, fr, NeSi_args.log_path, NeSi_args.wf_size, NeSi_args.gamma, NeSi_args.model_num)
        print(f'gamma:{NeSi_args.gamma}, Wf_size:{NeSi_args.wf_size}, Log path:{NeSi_args.log_path}, model:{fr}')

        set_seed(eval_config.config["yaml-config"]['env']['seed'])

        from config.train_set_config import trainSet_Generate
        yaml_path = 'config/workflow_scheduling_es_openai.yaml'
        train_Set_setting = trainSet_Generate(yaml_path)

        Builder(eval_config, train_Set_setting, test_Set_setting).build().eval()

if __name__ == "__main__":
    from config.test_set_config import testSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    test_Set_setting = testSet_Generate(yaml_path)

    NeSi_parser = argparse.ArgumentParser(description='settings func', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    NeSi_parser.add_argument('--gamma', '-g', required=True, help='the gamma')
    NeSi_parser.add_argument('--wf_size', '-w', required=True, help='the wf_size')
    NeSi_parser.add_argument('--log_path', '-log', required=True, help='the log')
    NeSi_parser.add_argument('--model_num', '-m', required=True, help='test on which model to evaluate, just for final mode')
    NeSi_args = NeSi_parser.parse_args()
    main(NeSi_args, test_Set_setting)
