from abc import *
import argparse
import yaml
from distutils.util import strtobool


class BaseConfig(metaclass=ABCMeta):

    def __init__(self, *args):
        parser = argparse.ArgumentParser(description='Arguments')
        parser.add_argument('--config', type=str, default='config/workflow_scheduling_es_openai.yaml',
                            help='A config path for env, policy, and optim')
        parser.add_argument('--processor_num', type=int, default=20, help='Specify processor number for multiprocessing')  # NOTE: We set this to 2 in our experiments
        parser.add_argument('--eval_ep_num', type=int, default=1, help='Set evaluation number per iteration')

        # Settings related to log
        # parser.add_argument("--log", action="store_true", help="Use log")
        parser.add_argument("--log", default=True, action="store_true", help="Use log")
        parser.add_argument('--save_model_freq', type=int, default=20, help='Save model every a few iterations')

        # Overwrite some common values in YAML with command-line options, if needed.
        parser.add_argument('--seed', type=int, default=None, help='Replace seed value in YAML')
        parser.add_argument('--reward', type=int, default=None, help='Select reward option')
        parser.add_argument('--sigma_init', type=float, default=None, help='Sigma init: noise standard deviation')
        parser.add_argument('--learning_rate', type=float, default=None, help='Replace learning rate in YAML')
        parser.add_argument('--reinforce_learning_rate', type=float, default=None, help='Replace reinforce learning rate in YAML')
        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            # Replace seed value if command-line options on seed is not None
            if args.seed is not None:
                config['env']['seed'] = args.seed

            if args.reward is not None:
                config['env']['reward'] = args.reward

            if args.sigma_init is not None:
                config['optim']['sigma_init'] = args.sigma_init

            if args.learning_rate is not None:
                config['optim']['learning_rate'] = args.learning_rate

            if args.reinforce_learning_rate is not None:
                config['optim']['reinforce_learning_rate'] = args.reinforce_learning_rate

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config

       
