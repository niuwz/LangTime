from exp.exp_basic import Exp_Basic
from utils.tools import (
    rm_ds_checkpints,
    extract_epoch_from_checkpoint_name,
)
from utils.metrics import metric
import torch
import os
import time
import warnings
import numpy as np
from data_provider.Mix_data_loader import MixDataLoader, EvalDataloader
from tqdm import tqdm
import random
import json

from transformers import AutoTokenizer
from configs.log_config import get_logger
from utils.plot_series import plot_all_result, plot_single_result
from shutil import rmtree

warnings.filterwarnings("ignore")
logger = get_logger()

class Exp_Eval(Exp_Basic):
    def __init__(self, args):
        self._build_tokenizer(args.backbone_path)
        self.use_ds = os.path.isfile(args.deepspeed_config)
        super(Exp_Eval, self).__init__(args)

    def _build_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "gpt" in model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ 'You are a helpful assistant.\n' }}{% endif %}{{message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '\n' }}{% endif %}"
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|ts_emb|>",
                    "<|TS_ENC|>",
                    "<|ts_mask|>",
                    "<|ts_out|>",
                ]
            }
        )
        self.placeholder: int = self.tokenizer("<|TS_ENC|>").input_ids[0]
        self.emb_token: int = self.tokenizer("<|ts_emb|>").input_ids[0]
        self.mask_token: int = self.tokenizer("<|ts_mask|>").input_ids[0]
        self.out_token: int = self.tokenizer("<|ts_out|>").input_ids[0]
        logger.debug(str(self.tokenizer))

    def _build_model(self):
        model = (
            self.model_dict[self.args.model]
            (self.args, self.placeholder, self.emb_token, self.mask_token, self.out_token, "ref_freeze")
            .bfloat16()
        )
        if self.args.model_init != "random":
            init_path = os.path.join(
                self.args.checkpoints, self.args.model_init, "checkpoint.pth"
            )
            logger.info("Init model by {}".format(init_path))
            model.load_state_dict(torch.load(init_path), strict=False)
        num_params = sum(p.numel() for p in model.parameters())
        train_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Number of parameters: {:,}".format(num_params))
        logger.debug(
            "Number of LLM parameters: {:,}".format(
                sum(
                    p.numel() for n, p in model.named_parameters() if "transformer" in n
                )
            )
        )
        logger.debug(
            "Number of Time Encoder parameters: {:,}".format(
                sum(p.numel() for n, p in model.named_parameters() if "ts_enc" in n)
            )
        )
        logger.info("Number of trained parameters: {:,}".format(train_num_params))
        logger.info(
            "Rate of trained parameters: {:.4f}%".format(
                train_num_params / num_params * 100
            )
        )
        logger.debug(str(model))

        return model

    def _get_data(self, flag):
        loaders = EvalDataloader(flag, self.args, self.tokenizer)
        return loaders

    def _apply_deepspeed(self, optimizer=None):
        if self.use_ds:
            import deepspeed
            from utils.tools import parser_deepspeed_config

            parameter = filter(lambda p: p.requires_grad, self.model.parameters())
            ds_config, ds_eval_config = parser_deepspeed_config(
                self.args.deepspeed_config, self.args
            )
            if optimizer is not None:
                self.model, optimizer, _, _ = deepspeed.initialize(
                    model=self.model,
                    config_params=ds_config,
                    model_parameters=parameter,
                    optimizer=optimizer,
                )
            else:
                self.model, _, _, _ = deepspeed.initialize(
                    model=self.model,
                    config_params=ds_eval_config,
                )
        else:
            self.model = self.model.to(self.device)
        return optimizer

    def _load_best_model_state(self, path):
        if self.use_ds:
            best_path = self._get_best_model_path(path)
            _, client_sd = self.model.load_checkpoint(os.path.join(path, best_path))
        else:
            self.model.load_state_dict(
                torch.load(path + "/checkpoint.pth"), strict=False
            )

    def _get_best_model_path(self, path):
        dirs = os.listdir(path)
        # Filter out checkpoints that did not improve
        improved_checkpoint_dirs = [i for i in dirs if i.startswith("checkpoint-epoch") and "-noimprove" not in i]
        
        if not improved_checkpoint_dirs:
            # If no improved checkpoints, fall back to all checkpoints for selection (with a warning)
            logger.warning("No improved checkpoints found. Falling back to all checkpoints for selection.")
            improved_checkpoint_dirs = [i for i in dirs if i.startswith("checkpoint-epoch")]

        improved_checkpoint_dirs.sort(key=extract_epoch_from_checkpoint_name)
        return improved_checkpoint_dirs[-1]

    def convert_best_model(self, path):
        if self.use_ds and self.args.local_rank == 0:
            from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

            logger.info(f"Start model checkpoint converting ...")
            best_path = self._get_best_model_path(path)
            convert_zero_checkpoint_to_fp32_state_dict(
                os.path.join(path, best_path), os.path.join(path, "checkpoint.pth")
            )
            del_paths = rm_ds_checkpints(path)
            logger.info(
                f"Deepspeed checkpoints: {del_paths} have been successfully removed."
            )
            logger.info(
                f"Model checkpoint has been successfully converted to {path}/checkpoint.pth."
            )

    def eval(self, setting):
        _ = self._apply_deepspeed()

        result_dict = {}
        test_loader = self._get_data(flag="test")
        apart_domain_idx = {
            test_loader.domain_list[i]: [i] * test_loader.batch_nums[i]
            for i in range(len(test_loader.domain_list))
        }
        self.model.eval()
        self.model.eval_mode()

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path + "images/"):
            os.makedirs(folder_path + "images/")

        infernce_time = 0

        for domain in apart_domain_idx.keys():
            domain_idx = apart_domain_idx[domain]
            reco_trues = []

            if "split" in domain:
                domain_name = domain.split("_")[0]+"_"+domain.split("_")[-1]
                result_dict[domain_name] = {
                    "mse": [],
                    "mae": [],
                }
            else:
                result_dict[domain] = {
                    "mse": [],
                    "mae": [],
                }

            preds = []
            pred_trues = []
            with torch.no_grad():
                test_loader.reset(domain_idx[0])
                if not (self.use_ds) or self.args.local_rank == 0:
                    steps_per_epoch = tqdm(
                        range(len(domain_idx)),
                        desc="Eval",
                        postfix={"domain": domain},
                        ncols=150,
                    )
                else:
                    steps_per_epoch = range(len(domain_idx))
                for i in steps_per_epoch:
                    prompt, mask, x, y, x_mark, y_mark = test_loader.get_batch(
                        domain_idx[i]
                    )
                    logger.debug(
                        self.tokenizer.batch_decode(prompt[0])
                    )
                    self.model.update_steps(y.shape[1]-self.args.label_len)

                    prompt = prompt
                    mask = mask
                    batch_x = x.bfloat16()
                    batch_y = y.bfloat16()
                    batch_x_mark = x_mark.bfloat16()
                    start_time = time.time()
                    if self.use_ds:
                        batch_x = batch_x.to(self.args.local_rank)
                        mask = mask.to(self.args.local_rank)
                        prompt = prompt.to(self.args.local_rank)
                        batch_x_mark = batch_x_mark.to(self.args.local_rank)
                        batch_y = batch_y[:, self.args.label_len:, :].to(self.args.local_rank)
                        y_mark = y_mark[:, self.args.label_len:, :].to(self.args.local_rank)
                    else:
                        batch_x = batch_x.to(self.device)
                        mask = mask.to(self.device)
                        prompt = prompt.to(self.device)
                        batch_x_mark = batch_x_mark.to(self.device)
                        batch_y = batch_y[:, self.args.label_len:, :].to(self.device)
                        y_mark = y_mark[:, self.args.label_len:, :].to(self.device)

                    outputs = self.model.generate(
                        batch_x, batch_x_mark, prompt, mask, y_mark
                    )
                    infernce_time += time.time() - start_time
                    f_dim = -1 if self.args.features == "MS" else 0
                    batch_x = batch_x[:, :, f_dim:]

                    batch_x = batch_x.detach().cpu().float().numpy()
                    reco_trues.append(batch_x)

                    pred_outputs = outputs[:, :batch_y.shape[1], f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    pred_outputs = pred_outputs.detach().cpu().float().numpy()
                    batch_y = batch_y.detach().cpu().float().numpy()

                    preds.append(pred_outputs)
                    pred_trues.append(batch_y)

            preds = np.concatenate(preds, axis=0)
            pred_trues = np.concatenate(pred_trues, axis=0)
            reco_trues = np.concatenate(reco_trues, axis=0)

            pred_mae, pred_mse, pred_rmse, pred_mape, pred_mspe, pred_cr = metric(
                preds, pred_trues
            )
            logger.info(
                "[Rank-{}] | [Prection] domain:{}, mse:{}, mae:{}".format(
                    self.args.local_rank, domain, pred_mse, pred_mae
                )
            )

            if "split" in domain:
                domain_name = domain.split("_")[0]+"_"+domain.split("_")[-1]
                result_dict[domain_name]["mse"].append(pred_mse)
                result_dict[domain_name]["mae"].append(pred_mae)
            else:
                result_dict[domain]["mse"].append(pred_mse)
                result_dict[domain]["mae"].append(pred_mae)

            np.save(folder_path + domain + f"_predicton_rank{self.args.local_rank}.npy", preds)
            np.save(folder_path + domain + f"_pred_true_rank{self.args.local_rank}.npy", pred_trues)
            np.save(folder_path + domain + f"_reco_true_rank{self.args.local_rank}.npy", reco_trues)

            if hasattr(self.args, "plot_img") and self.args.plot_img:
                plot_all_result(
                    preds,
                    pred_trues,
                    folder_path + "images/" + domain + "_prediction_all.png",
                )

                for c in range(preds.shape[-1]):
                    for p in range(0, preds.shape[0], 1000):
                        plot_single_result(
                            preds[p, :, c],
                            pred_trues[p, :, c],
                            folder_path
                            + "images/"
                            + domain
                            + f"_prediction_{p}_channel_{c}_rank{self.args.local_rank}.png",
                            input_series=reco_trues[p, :, c],
                        )
                logger.info(f"Result images have been saved to {folder_path}images/")

        for domain_name in result_dict.keys():
            result_dict[domain_name]["mse"] = np.mean(result_dict[domain_name]["mse"])
            result_dict[domain_name]["mae"] = np.mean(result_dict[domain_name]["mae"])
        
        world_size = torch.distributed.get_world_size() if self.use_ds else 1
        gathered_results = [None for _ in range(world_size)]
        if self.use_ds:
            torch.distributed.all_gather_object(gathered_results, result_dict)
        else:
            gathered_results[0] = result_dict

        if self.args.local_rank == 0:
            final_results = {}
            for res in gathered_results:
                for domain_name in res.keys():
                    if domain_name not in final_results:
                        final_results[domain_name] = {"mse": [], "mae": []}
                    final_results[domain_name]["mse"].append(res[domain_name]["mse"])
                    final_results[domain_name]["mae"].append(res[domain_name]["mae"])
            
            for domain_name in final_results.keys():
                final_results[domain_name]["mse"] = float(np.mean(final_results[domain_name]["mse"]))
                final_results[domain_name]["mae"] = float(np.mean(final_results[domain_name]["mae"]))
            
            logger.info("All Rank Results:")
            for domain_name, metrics in final_results.items():
                logger.info(f"Domain: {domain_name}, MSE: {metrics['mse']}, MAE: {metrics['mae']}")
            
            logger.info(f"Total inference time: {infernce_time} seconds")
            logger.info(f"Average inference time per sample: {infernce_time / len(domain_idx)} seconds")
            
            final_results["in_len"] = self.args.seq_len
            final_results["model"] = self.args.model_init
            os.makedirs(os.path.join(folder_path, "metrics"), exist_ok=True)
            with open(os.path.join(folder_path, "metrics", "eval_results.json"), "w") as f:
                json.dump(final_results, f, indent=4)