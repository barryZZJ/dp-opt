import argparse
import multiprocessing
import datetime
import socket
import os

from dpopt.experiments.exp_default import run_exp_default
from dpopt.utils.torch import torch_initialize
from dpopt.utils.my_logging import log
from dpopt.search.dpconfig import DPConfig

if __name__ == "__main__":
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    hostname = socket.gethostname()
    log_tag = timestamp + '_' + hostname
    log_dir = os.path.join("logs", log_tag)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=log_dir, help='directory for output data')
    parser.add_argument('--processes', type=int, help='number of processes to use', default=multiprocessing.cpu_count())
    parser.add_argument('--torch-device', default='cpu', help='the pytorch device use (untested on GPU!)')
    parser.add_argument('--torch-threads', default=8, help='the number of threads for pytorch to use')
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    n_processes = args.processes
    torch_initialize(args.torch_threads, args.torch_device)
    log.configure("INFO")

    config = DPConfig(n_processes=n_processes)
    if args.test:
        config = DPConfig(n_train=10, n=10, n_final=10, n_processes=2)
    # run experiments on all algorithms using logistic regression classifier
    run_exp_default("power_search_reg", args.output_dir, config)
