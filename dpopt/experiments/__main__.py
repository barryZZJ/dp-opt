import argparse
import datetime
import multiprocessing
import os
import socket

from dpopt.experiments.exp_baseline import run_exp_baseline
from dpopt.experiments.exp_default import run_exp_default
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log
from dpopt.utils.torch import torch_initialize

if __name__ == "__main__":
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    hostname = socket.gethostname()
    log_tag = timestamp + '_' + hostname
    log_dir = os.path.join("logs", log_tag)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-baseline', action="store_true", help='run the baseline (DP-Sniper)')
    parser.add_argument('--output-dir', default=log_dir, help='directory for output data')
    parser.add_argument('--torch-device', default='cpu', help='the pytorch device use (untested on GPU!)')
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    torch_initialize(torch_threads=None, torch_device=args.torch_device)
    log_level = 'INFO'
    file_level = 'INFO'

    config = DPConfig()
    if args.test:
        # config = DPConfig(n_train=10, n_check=10, n_final=10)
        config = DPConfig(n_train=20000, n_check=20000, n_final=20000)
        args.output_dir += '_test'

    if args.run_baseline:
        log.info("running baseline for comparison")
        args.output_dir += '_baseline'
        run_exp_baseline("DDSearch_reg", args.output_dir, config, log_level, file_level)
    else:
        log.info("running PowerSearcher")
        run_exp_default("PowerSearcher_reg", args.output_dir, config, log_level, file_level)
