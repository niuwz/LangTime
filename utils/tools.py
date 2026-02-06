import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
import os
import json
import torch.distributed as dist
from shutil import rmtree

import math
import shutil
from configs.log_config import get_logger

logger = get_logger()
plt.switch_backend("agg")


def save_configs(args, save_name):
    if args.local_rank > 0:
        return ""
    save_path = os.path.join(args.checkpoints, save_name)
    if os.path.exists(save_path):
        rmtree(save_path)
    os.makedirs(save_path)

    config_dict = dict(vars(args))
    config_dict["backbone_config"] = vars(config_dict["backbone_config"])
    if os.path.isfile(config_dict["deepspeed_config"]):
        config_dict["deepspeed_config"] = parser_deepspeed_config(
            config_dict["deepspeed_config"], args
        )
    filename = os.path.join(save_path, "config.json")
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, ensure_ascii=False, indent=4)
    return f"Arguments have been saved to '{filename}'."


def parser_deepspeed_config(ds_config_path, args):
    import deepspeed

    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    world_size = deepspeed.comm.get_world_size()
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.gradient_clipping
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = (
        args.batch_size * world_size * args.gradient_accumulation_steps
    )
    ds_eval_config = get_eval_ds_config(
        offload=True,
        dtype="bf16",
        micro_batch_size=ds_config["train_micro_batch_size_per_gpu"],
        global_batch_size=ds_config["train_batch_size"],
        gradient_clipping=ds_config["gradient_clipping"],
    )
    return ds_config, ds_eval_config


def get_eval_ds_config(
    offload, dtype, micro_batch_size, global_batch_size, gradient_clipping=1.0, stage=0
):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {"device": device},
        "memory_efficient_linear": False,
    }
    return {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": gradient_clipping,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def temperature_softmax(x, temperature=1.0, dim=-1):
    """
    Applies the softmax function with a temperature parameter to the input tensor.

    Args:
    - x (torch.Tensor): Input tensor.
    - temperature (float): Temperature parameter. Higher values make the distribution more uniform, while lower values sharpen it.

    Returns:
    - torch.Tensor: Softmax output with the given temperature applied.
    """
    # Avoid numerical instability by subtracting the maximum value from each element
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_shifted = x - x_max

    # Apply temperature scaling
    x_scaled = x_shifted / temperature

    # Compute the exponentials
    exp_x_scaled = torch.exp(x_scaled)

    # Normalize by the sum of exponentials
    softmax_output = exp_x_scaled / torch.sum(exp_x_scaled, dim=dim, keepdim=True)

    return softmax_output


def adjust_learning_rate_per_step(step, optimizer, args, total_steps):
    if step < args.warmup_steps:
        alpha = step / args.warmup_steps
        lr = args.initial_lr * alpha
    else:
        adjusted_step = step - args.warmup_steps
        if adjusted_step >= total_steps - args.warmup_steps:
            lr = args.min_lr
        else:
            lr = (
                args.min_lr
                + (args.initial_lr - args.min_lr)
                * (
                    1
                    + math.cos(
                        math.pi * adjusted_step / (total_steps - args.warmup_steps)
                    )
                )
                / 2
            )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_annealing_with_warmup(
    initial_lr, min_lr, total_steps, warmup_steps, use_ds=False
):


    def scheduler(step, optimizer_or_model):
        if step < warmup_steps:
            alpha = step / warmup_steps
            lr = initial_lr * alpha
        else:
            adjusted_step = step - warmup_steps
            if adjusted_step >= total_steps - warmup_steps:
                lr = min_lr
            else:
                lr = (
                    min_lr
                    + (initial_lr - min_lr)
                    * (
                        1
                        + math.cos(
                            math.pi * adjusted_step / (total_steps - warmup_steps)
                        )
                    )
                    / 2
                )
        if use_ds:
            for param_group in optimizer_or_model.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            for param_group in optimizer_or_model.param_groups:
                param_group["lr"] = lr
        return lr

    return scheduler


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if args.lradj == "type7":
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    if args.lradj == "type6":
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "warmup_linear":
        lr_adjust = {2: 1e-3}
        if epoch > 2:
            lr_adjust.update({epoch: 1e-3 - 5e-5 * (epoch - 2) // 2})
    elif args.lradj == "warmup_cos":
        lr_adjust = {2: 1e-3}
        if epoch > 2:
            lr_adjust.update({epoch: 1e-3 - 5e-5 * (epoch - 2) // 2})
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, use_ds=False, epoch=0):
        es_dicision = self.decision(val_loss)
        if es_dicision:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, path, use_ds, epoch)
            self.counter = 0

    def decision(self, val_loss):
        """
        True: early stop +1
        False: 
        """
        val_flag = False
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            val_flag = False
        elif score < self.best_score + self.delta:
            val_flag = True
        else:
            self.best_score = score
            val_flag = False
        return val_flag

    def save_checkpoint(self, val_loss, model, path, use_ds, epoch):
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if use_ds:
            model.save_checkpoint(path + "/" + f"checkpoint-epoch{epoch}")
        else:
            torch.save(model.fine_tune_state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def rm_ds_checkpints(path, res_nums=0):
    dirs = os.listdir(path)
    checkpoint_dirs = [i for i in dirs if i.startswith("checkpoint-epoch")]
    del_paths = []
    if len(checkpoint_dirs) > res_nums:
        checkpoint_dirs.sort(key=lambda x: int(x[len("checkpoint-epoch") :]))
        if res_nums > 0:
            checkpoint_dirs = checkpoint_dirs[:-res_nums]
        for d in checkpoint_dirs:
            dir_path = os.path.join(path, d)
            del_paths.append(d)
            shutil.rmtree(dir_path)
    return del_paths


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
