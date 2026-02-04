import warnings
import numpy as np
import random
from exp.exp_langtime import Exp_Mix
import torch
import argparse
import os
import time
import json
from utils.tools import save_configs
from configs.configs import PeftQwenConfig, get_args
from configs.log_config import setup_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

import deepspeed
deepspeed.init_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeQwen")
    args = get_args(parser)

    logger = setup_logging(args.log_file, local_rank=args.local_rank)

    args.use_gpu = args.use_gpu if torch.cuda.is_available() else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = eval(args.devices)

    logger.info(vars(args))
    Exp = Exp_Mix

    with open(args.config_path, "r") as f:
        args.domains = json.load(f)
    if args.domain[0].lower() != "mix":
        args.domains = {k: args.domains[k] for k in args.domain}
    args.backbone_config = PeftQwenConfig(
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.dropout
    )

    domain = "_".join(args.domains.keys())
    origin_setting = f"{args.save_name}_{args.task_id}"
    if args.is_training:
        logger.info(save_configs(args, origin_setting))
        for ii in range(args.itr):
            run_seed = int(random.random() * 10000)
            logger.info(f"seed: {run_seed}")
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            setting = origin_setting

            exp = Exp(args)  # set experiments
            logger.info(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)
            logger.info(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 1
        setting = origin_setting + str(ii)

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
