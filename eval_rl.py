import numpy as np
import os
import random
import torch
from builder import Builder
from config.eval_config import EvalConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(test_Set_setting):
    # which_log = 'logs/WorkflowScheduling-v0'
    which_log = 'logs/WorkflowScheduling-v3'
    log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]

    Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
    WF_Size = ['S', 'M', 'L']
    for gamma in Gamma:  # Delete all existed files before we run the testing code
        for wf_size in WF_Size:
            for log_path in log_folders:
                dir_csv = log_path + "/test_performance" + "/testing_record_" + str(gamma) + "_" + str(wf_size) + ".csv"
                if os.path.exists(dir_csv):
                    os.remove(dir_csv)

    for log_path in log_folders:
        for fr in np.arange(0, 3020, 20, dtype=int):
            # log file not exits, break the loop
            model = f'{log_path}/saved_models/ep_{fr}.pt'
            if not os.path.exists(model):
                break
            eval_config = EvalConfig(fr, log_path)
            set_seed(eval_config.config["yaml-config"]['env']['seed'])

            # Create a training set under current seed
            from config.train_set_config import trainSet_Generate
            yaml_path = 'config/workflow_scheduling_es_openai.yaml'
            train_Set_setting = trainSet_Generate(yaml_path)

            Builder(eval_config, train_Set_setting, test_Set_setting).build().eval()


if __name__ == "__main__":
    # TODO:
    # Default gamma = 1.0 and WF Size = S in yaml.
    # You should specify the gamma and WF size to collect the corresponding testing results

    # Create the fixed testing set
    from config.test_set_config import testSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    test_Set_setting = testSet_Generate(yaml_path)

    main(test_Set_setting)

