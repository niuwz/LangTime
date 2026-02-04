from transformers.models.qwen2 import Qwen2Config
import argparse
from typing import Literal


class PeftQwenConfig(Qwen2Config):
    def __init__(self, lora_rank=32, lora_alpha=32.0, lora_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        # LoRA
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def get_args(parser: argparse.ArgumentParser, mode: Literal["pt", "rl", "eval"] = "pt"):
    """
    mode: pt-pretrain, rl, or eval
    """
    # Task
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument(
        "--model",
        type=str,
        default="langtime",
        help="model name, options: [langtime, langtime_eval, langtime_rl]",
    )
    parser.add_argument("--task_id", type=str, default="pt")
    parser.add_argument("--save_name", type=str, default="qwen")
    parser.add_argument(
        "--model_init", type=str, default="random", help="random or model path"
    )

    # input-output
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--single_pred_len", type=int, default=96, help="prediction sequence length"
    )

    # base
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument("--train_epochs", type=int, default=5, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="batch size of eval input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--initial_lr", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--lr_decay", type=float, default=0.01, help="")
    parser.add_argument("--warmup_rate", type=float, default=0.05, help="")
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=[0.3,],
        nargs="+",
        help="",
    )
    parser.add_argument(
        "--loss_alpha_type",
        type=str,
        default="fix",
        choices=["fix", "reduce"],
        help="fix or reduce",
    )
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--huber_delta", type=float, default=0.5, help="loss function")

    parser.add_argument(
        "--use_amp", action="store_true", help="use automatic mixed precision training"
    )
    parser.add_argument(
        "--training_mode", type=str, default="full", help="training_mode"
    )

    # model
    parser.add_argument("--backbone", type=str, default="qwen2", help="")
    parser.add_argument(
        "--backbone_path",
        type=str,
        required=True,
        default="",
        help="",
    )
    parser.add_argument("--use_flash_attn", action="store_true", help="")
    parser.add_argument("--d_model", type=int, default=896, help="")

    # PEFT
    parser.add_argument("--lora_rank", type=int, default=32, help="")
    parser.add_argument("--lora_alpha", type=int, default=32, help="")
    parser.add_argument("--dropout", type=int, default=0.1, help="")

    # TimeEncoder
    parser.add_argument("--ts_enc", type=str, default="patch", help="")
    parser.add_argument("--d_ff", type=int, default=4864, help="")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--factor", type=int, default=1, help="")
    parser.add_argument("--e_layers", type=int, default=1, help="")
    parser.add_argument("--output_attention", action="store_true", help="")
    parser.add_argument("--n_heads", type=int, default=8, help="")
    parser.add_argument("--num_kv_heads", type=int, default=2, help="")
    # patch
    parser.add_argument("--patch_size", type=int, default=16, help="")

    # Q-Former
    parser.add_argument("--q_layers", type=int, default=1, help="layer of q-former")
    parser.add_argument(
        "--q_num", type=int, default=1, help="num of tarinable qureies in q-former"
    )
    parser.add_argument(
        "--adapter_type", type=str, default="linear", help="t_former, q_former, linear"
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        nargs="+",
        help="single domain: select in [ETTh1, ETTh2, ETTm1, ETTm2, Weather]\nmulti domain: [domain1, domain2]\nall domain: mix or Mix",
    )
    parser.add_argument("--inverse", action="store_true", help="")
    parser.add_argument(
        "--features", type=str, default="M", help="select in [M, S, MS]"
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--enc_mask",
        type=str,
        default="no",
        help="time series mask, options:[no, fix:n]",
    )

    # MixDataloader
    parser.add_argument(
        "--data_description_path",
        type=str,
        default="configs/data_description.json",
        help="",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/datasets_config.json",
        help="path of dataset description",
    )
    parser.add_argument("--data_dir", type=str, default="./datasets/", help="")
    parser.add_argument("--num_workers", type=int, default=10, help="")
    parser.add_argument("--scale", action="store_false", help="")
    parser.add_argument("--percent", type=int, default=100, help="")
    parser.add_argument("--split", type=str, default=":", help="")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="single gpu id")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus"
    )
    parser.add_argument(
        "--devices", type=str, default="[0,1]", help="device ids of multile gpus"
    )

    # deepspeed
    parser.add_argument("--deepspeed_config", type=str, default="", help="")
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--gradient_clipping",
        type=int,
        default=1.0,
        help="gradient_clipping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )

    # other
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/run.log",
        help="fixed, time or custom file name",
    )
    if mode == "rl":
        # PPO
        parser.add_argument("--upd_epochs", type=int, default=3, help="")
        parser.add_argument("--kl_ctl", type=float, default=0.1, help="")
        parser.add_argument("--tau", type=float, default=0.1, help="")
        parser.add_argument("--clip_reward_value", type=float, default=5.0, help="")
        parser.add_argument("--cliprange", type=float, default=0.2, help="")
        parser.add_argument("--cliprange_value", type=float, default=0.2, help="")
        parser.add_argument("--gamma", type=float, default=1.0, help="")
        parser.add_argument("--lam", type=float, default=0.95, help="")
        parser.add_argument("--err_ctrl", type=float, default=0.7, help="")
        parser.add_argument("--adv_norm", action="store_true", help="")
    elif mode == "pt":
        parser.add_argument(
            "--pretrain_seq_lens",
            type=int,
            required=True,
            nargs="+",
            default=[96,],
            help="",
        )

    elif mode == "eval":
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default="",
            help="custom file name",
        )
        parser.add_argument(
            "--tasks_lens",
            type=int,
            required=True,
            nargs="+",
            default=[96, 192, 336, 720],
            help="",
        )

    args = parser.parse_args()
    return args
