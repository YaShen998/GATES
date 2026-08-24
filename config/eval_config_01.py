from abc import *
import argparse
import yaml
import os


class EvalConfig(metaclass=ABCMeta):
    def __init__(self, *args):
        model_count, fr, log_path, wf_size, gamma, model_num = args
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=f"{log_path}/profile.yaml")
        parser.add_argument("--policy_path", type=str, default=f'{log_path}/saved_models/ep_{fr}.pt', help='saved model directory')
        parser.add_argument("--rms_path", type=str, default=f'{log_path}/saved_models/ob_rms_{fr}.pickle', help='saved run-time mean and std directory')
        parser.add_argument('--eval_ep_num', type=int, default=1, help='Set evaluation number per iteration')
        parser.add_argument('--save_model_freq', type=int, default=20, help='Save model every a few iterations')
        parser.add_argument('--processor_num', type=int, default=1, help='Testing model only use 1 processor')

        parser.add_argument("--log", default=True, action="store_true", help="Use log")
        parser.add_argument("--save_gif", action="store_true")

        parser.add_argument('--wf_size', '-w', type=str, default=wf_size)
        parser.add_argument('--gamma_test', '-g', type=float, default=gamma)
        parser.add_argument('--log_path', '-log', type=str, default=log_path)
        parser.add_argument('--model_num', '-m', type=int, default=model_num)
        parser.add_argument('--final', '-f', type=int, default=False)
        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            if args.wf_size is not None:
                config['env']['wf_size'] = args.wf_size

            if args.gamma_test is not None:
                config['env']['gamma_test'] = args.gamma_test

            if args.gamma_test is not None:
                config['env']['model_num'] = args.model_num

            if fr == model_num and model_count == 1:
                args.final = True
                print(f"Only test the final episode model: {args.final}")

        if args.save_gif:
            run_num = args.ckpt_path.split("/")[-3]
            save_dir = f"test_gif/{run_num}/"
            os.makedirs(save_dir)

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config
