from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    cosine_annealing_with_warmup,
    rm_ds_checkpints,
    extract_epoch_from_checkpoint_name,
)
from utils.metrics import metric
from utils.losses import langtime_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from data_provider.Mix_data_loader import MixDataLoader, PreTrainDataloader
from tqdm import tqdm
import random

from transformers import AutoTokenizer
from configs.log_config import get_logger
from utils.plot_series import plot_all_result, plot_single_result
from collections import defaultdict

warnings.filterwarnings("ignore")
logger = get_logger()


class Exp_Mix(Exp_Basic):
    def __init__(self, args):
        self._build_tokenizer(args.backbone_path)
        self.use_ds = os.path.isfile(args.deepspeed_config)
        super(Exp_Mix, self).__init__(args)

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
            self.model_dict[self.args.model](self.args, self.placeholder, self.emb_token, self.mask_token, self.out_token)
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
        logger.info(f"Loading {flag} data... This may take a while, please be patient.")
        loaders = PreTrainDataloader(flag, self.args, self.tokenizer)
        return loaders

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), weight_decay=0.1)
        return model_optim

    def _select_criterion(self):
        criterion = langtime_loss(
            alpha=self.args.loss_alpha,
            alpha_type=self.args.loss_alpha_type,
            loss=self.args.loss,
            huber_delta=self.args.huber_delta,
        )
        return criterion

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
            # If no improved checkpoints, fall back to all checkpoints (user might want to pick manually)
            logger.warning("No improved checkpoints found. Falling back to all checkpoints for selection.")
            improved_checkpoint_dirs = [i for i in dirs if i.startswith("checkpoint-epoch")]
            
        improved_checkpoint_dirs.sort(key=extract_epoch_from_checkpoint_name)
        return improved_checkpoint_dirs[-1]

    def convert_best_model(self, path):
        if self.use_ds and self.args.local_rank == 0:
            from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

            logger.info("Start model checkpoint converting ...")
            best_path = self._get_best_model_path(path)
            convert_zero_checkpoint_to_fp32_state_dict(
                os.path.join(path, best_path), os.path.join(path, "checkpoint.pth")
            )
            del_paths = rm_ds_checkpints(path, convert_before_del=True)
            logger.info(
                f"Deepspeed checkpoints: {del_paths} have been successfully removed."
            )
            logger.info(
                f"Model checkpoint has been successfully converted to {path}/checkpoint.pth."
            )

    def domain_shuffle(self, dataloader, current_step=0, warmup_steps=0):
        total_epoch_batches = sum(dataloader.batch_nums)
        remaining_global_warmup = max(0, warmup_steps - current_step)
        epoch_warmup_steps = min(remaining_global_warmup, total_epoch_batches)

        if epoch_warmup_steps == 0:
            full_batches = []
            for i, num_batches in enumerate(dataloader.batch_nums):
                full_batches.extend([i] * num_batches)
            random.shuffle(full_batches)
            return full_batches
        
        warmup_subsets = []
        annealing_pool = []
        warmup_ratio = epoch_warmup_steps / total_epoch_batches

        for i, domain_name in enumerate(dataloader.domain_list):
            num_batches = dataloader.batch_nums[i]
            if num_batches == 0:
                continue
            n_warmup = max(1, int(num_batches * warmup_ratio))

            all_domain_batches = [i] * num_batches
            random.shuffle(all_domain_batches)

            warmup_part = all_domain_batches[:n_warmup]
            annealing_part = all_domain_batches[n_warmup:]
            length = int(domain_name.split('_')[-1])
            warmup_subsets.append({'length': length, 'batches': warmup_part})
            annealing_pool.extend(annealing_part)
        warmup_subsets.sort(key=lambda x: x['length'])
        
        warmup_pool = []
        for subset in warmup_subsets:
            warmup_pool.extend(subset['batches'])
        random.shuffle(annealing_pool)
        random.shuffle(warmup_pool)
        return warmup_pool + annealing_pool

    def domain_mask_rate(self, train_loader, epoch=None, current_step=None):
        if self.args.enc_mask == "no":
            domain_mask = {k: 0.0 for k in train_loader.domain_list}
        elif self.args.enc_mask.startswith("fix"):
            fixed_rate = float(self.args.enc_mask.split(":")[1])
            domain_mask = {k: fixed_rate for k in train_loader.domain_list}
        elif self.args.enc_mask == "anneal":
            if current_step is not None:
                total_steps = self.args.train_epochs * len(train_loader)
                mask_rate = max(0.0, 0.5 - (current_step / total_steps))
            else:
                mask_rate = 0.5
            domain_mask = {k: mask_rate for k in train_loader.domain_list}
        else:
            raise ValueError(f"Unsupported enc_mask type {self.args.enc_mask}")
        return domain_mask

    def train(self, setting):
        train_loader = self._get_data(flag="train")
        vali_loader = self._get_data(flag="val")
        test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        model_optim = self._select_optimizer()
        model_optim = self._apply_deepspeed(model_optim)
        criterion = self._select_criterion()

        # cosine_annealing_with_warmup
        total_steps = train_steps * self.args.train_epochs
        warmup_steps = train_steps * int(self.args.warmup_rate)
        min_lr = self.args.initial_lr * self.args.lr_decay
        lr_scheduler = cosine_annealing_with_warmup(
            self.args.initial_lr, min_lr, total_steps, warmup_steps
        )

        domain_idx = self.domain_shuffle(train_loader, current_step=0, warmup_steps=warmup_steps)
        logger.debug(
            self.tokenizer.batch_decode(train_loader.get_batch(domain_idx[0])[0][0])
        )

        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            model_optim.zero_grad()
            train_loss = []
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            current_epoch_start_step = epoch * train_steps
            domain_idx = self.domain_shuffle(train_loader, current_step=current_epoch_start_step, warmup_steps=warmup_steps)
            train_loader.reset(epoch=epoch)
            if not (self.use_ds) or self.args.local_rank == 0:
                steps_per_epoch = tqdm(
                    range(len(domain_idx)),
                    desc="Train",
                    ncols=150,
                    postfix={"epoch": epoch + 1},
                )
            else:
                steps_per_epoch = range(len(domain_idx))
            for i in steps_per_epoch:
                current_step = i + epoch * train_steps + 1
                current_lr = lr_scheduler(current_step, model_optim)
                if current_step > warmup_steps:
                    criterion.set_alpha(self.args.loss_alpha[1])
                prompt, mask, x, y, x_mark, y_mark = train_loader.get_batch(
                    domain_idx[i]
                )
                # get mask rate
                domain_mask = self.domain_mask_rate(train_loader, epoch, current_step)
                ts_mask_rate = domain_mask[train_loader.domain_list[domain_idx[i]]]
                iter_count += 1
                prompt = prompt
                mask = mask
                batch_x = x.bfloat16()
                batch_y = y.bfloat16()
                batch_x_mark = x_mark.bfloat16()
                seq_len = batch_x.shape[1]

                if self.use_ds:
                    batch_x = batch_x.to(self.model.local_rank)
                    mask = mask.to(self.model.local_rank)
                    prompt = prompt.to(self.model.local_rank)
                    batch_x_mark = batch_x_mark.to(self.model.local_rank)
                    batch_y = batch_y.to(self.model.local_rank)
                    outputs = self.model(
                        batch_x, batch_x_mark, prompt, mask, ts_mask_rate
                    )
                    f_dim = -1 if self.args.features == "MS" else 0
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(
                        outputs[:, : seq_len, :],
                        outputs[:, -self.args.pred_len :, f_dim:],
                        batch_x[:, :, f_dim:],
                        batch_y,
                    )
                    train_loss.append(loss.item())
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, prompt, mask, ts_mask_rate
                    )

                    f_dim = -1 if self.args.features == "MS" else 0
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(
                        outputs[:, : seq_len, :],
                        outputs[:, -self.args.pred_len :, f_dim:],
                        batch_x[:, :, f_dim:],
                        batch_y,
                    )
                    train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    logger.debug(
                        "iters: {}, epoch: {} | loss: {:.7f} | learning rate: {}".format(
                            i + 1, epoch + 1, loss.item(), current_lr
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    logger.debug(
                        "speed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.use_ds:
                    self.model.backward(loss)
                    if (
                        (current_step + 1) % self.args.gradient_accumulation_steps == 0
                        or total_steps == current_step + 1
                    ):
                        self.model.step()
                        model_optim.zero_grad()
                else:
                    loss.backward()
                    model_optim.step()
            if len(domain_idx) % self.args.gradient_accumulation_steps != 0:
                self.model.step()
                model_optim.zero_grad()

            logger.info(
                "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            )
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = self.vali(vali_loader, criterion, pred_loss_only=True)
            test_loss, test_metrics = self.vali(test_loader, criterion, pred_loss_only=True)
            logger.info(
                "Epoch: {}, Steps: {} | Train Loss: {:.4f}".format(
                    epoch + 1, train_steps, train_loss
                )
            )
            avg_loss = sum(vali_loss.values()) / len(vali_loss.values())
            logger.info(
                "Vali Loss: ["
                + " ".join([f"{k}: {v:.6f}" for k, v in vali_loss.items()])
                + "] | Avg Vali Loss: {:.6f}".format(avg_loss)
            )
            avg_test_loss = sum(test_loss.values()) / len(test_loss.values())
            logger.info(
                "Test Loss: ["
                + " ".join([f"{k}: {v:.6f}" for k, v in test_loss.items()])
                + "] | Avg Test Loss: {:.6f}".format(avg_test_loss)
            )

            logger.info(
                "Vali Reco Metircs: ["
                + " ".join(
                    [
                        f"{k}: mae-{v['reco'][0]:.6f},mse-{v['reco'][1]:.6f}"
                        for k, v in vali_metrics.items()
                    ]
                )
                + "]"
            )
            logger.info(
                "Vali Pred Metircs: ["
                + " ".join(
                    [
                        f"{k}: mae-{v['pred'][0]:.6f},mse-{v['pred'][1]:.6f}"
                        for k, v in vali_metrics.items()
                    ]
                )
                + "]"
            )

            logger.info(
                "Test Reco Metircs: ["
                + " ".join(
                    [
                        f"{k}: mae-{v['reco'][0]:.6f},mse-{v['reco'][1]:.6f}"
                        for k, v in test_metrics.items()
                    ]
                )
                + "]"
            )
            logger.info(
                "Test Pred Metircs: ["
                + " ".join(
                    [
                        f"{k}: mae-{v['pred'][0]:.6f},mse-{v['pred'][1]:.6f}"
                        for k, v in test_metrics.items()
                    ]
                )
                + "]"
            )

            # mask
            logger.info(
                "Mask Rate: ["
                + " ".join([f"{k}: {v}" for k, v in domain_mask.items()])
                + "]"
            )

            early_stopping(avg_loss, self.model, path, self.use_ds, epoch)
            if early_stopping.early_stop:
                logger.warning("Early stopping")
                break

        self._load_best_model_state(path)
        self.convert_best_model(path)
        return self.model

    def __eval_one_domain(self, data_loader: MixDataLoader, domain_idx, criterion):
        total_loss = []

        recos = []
        reco_trues = []
        preds = []
        pred_trues = []

        with torch.no_grad():
            data_loader.reset(domain_idx[0])
            if not (self.use_ds) or self.args.local_rank == 0:
                steps_per_epoch = tqdm(
                    range(len(domain_idx)),
                    desc=data_loader.flag.capitalize(),
                    postfix={"domain": data_loader.domain_list[domain_idx[0]]},
                    ncols=150,
                )
            else:
                steps_per_epoch = range(len(domain_idx))
            for i in steps_per_epoch:
                prompt, mask, x, y, x_mark, y_mark = data_loader.get_batch(
                    domain_idx[i]
                )
                prompt = prompt
                mask = mask
                batch_x = x.bfloat16()
                batch_y = y.bfloat16()
                batch_x_mark = x_mark.bfloat16()
                seq_len = batch_x.shape[1]

                if self.use_ds:
                    batch_x = batch_x.to(self.model.local_rank)
                    mask = mask.to(self.model.local_rank)
                    prompt = prompt.to(self.model.local_rank)
                    batch_x_mark = batch_x_mark.to(self.model.local_rank)
                    batch_y = batch_y.to(self.model.local_rank)
                outputs = self.model(batch_x, batch_x_mark, prompt, mask)
                f_dim = -1 if self.args.features == "MS" else 0

                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                loss = criterion(
                    outputs[:, : seq_len, :],
                    outputs[:, -self.args.pred_len :, f_dim:],
                    batch_x,
                    batch_y,
                )
                total_loss.append(loss.cpu().item())

                pred = outputs.detach().cpu().float().numpy()
                true = batch_y.detach().cpu().float().numpy()
                reconstruction = batch_x.detach().cpu().float().numpy()

                preds.append(pred[:, -self.args.pred_len :, f_dim:])
                pred_trues.append(true)

                recos.append(pred[:, : seq_len, :])
                reco_trues.append(reconstruction)

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        pred_trues = np.concatenate(pred_trues, axis=0)

        recos = np.concatenate(recos, axis=0)
        reco_trues = np.concatenate(reco_trues, axis=0)
        pred_mae, pred_mse, pred_rmse, pred_mape, pred_mspe, _ = metric(
            preds, pred_trues
        )
        reco_mae, reco_mse, reco_rmse, reco_mape, reco_mspe, _ = metric(
            recos, reco_trues
        )
        metrics = {"reco": (reco_mae, reco_mse), "pred": (pred_mae, pred_mse)}
        return total_loss, metrics

    def vali(self, vali_loader, criterion, pred_loss_only=False):
        self.model.eval()
        vali_losses = {}
        vali_metrics = {}
        apart_domain_idx = {
            vali_loader.domain_list[i]: [i] * vali_loader.batch_nums[i]
            for i in range(len(vali_loader.domain_list))
        }

        if pred_loss_only:
            orinal_alpha = criterion.alpha
            criterion.set_alpha(0.0)

        for k in apart_domain_idx.keys():
            domain_idx = apart_domain_idx[k]
            vali_loss_per_domain, vali_metrics_per_domain = self.__eval_one_domain(
                vali_loader, domain_idx, criterion
            )
            vali_losses.update({k: vali_loss_per_domain})
            vali_metrics.update({k: vali_metrics_per_domain})

        if pred_loss_only:
            criterion.set_alpha(orinal_alpha)
        return vali_losses, vali_metrics

    def test(self, setting, test=0, generate=False):
        test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")),
                strict=False,
            )
            self._apply_deepspeed()
        apart_domain_idx = {
            test_loader.domain_list[i]: [i] * test_loader.batch_nums[i]
            for i in range(len(test_loader.domain_list))
        }
        self.model.eval()

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path + "images/"):
            os.makedirs(folder_path + "images/")

        for domain in apart_domain_idx.keys():
            domain_idx = apart_domain_idx[domain]
            reco = []
            reco_trues = []
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
                    prompt = prompt.to(self.device)
                    mask = mask.to(self.device)
                    batch_x = x.bfloat16().to(self.device)
                    batch_y = y.bfloat16().to(self.device)
                    batch_x_mark = x_mark.bfloat16().to(self.device)
                    seq_len = batch_x.shape[1]

                    if self.use_ds:
                        batch_x = batch_x.to(self.args.local_rank)
                        mask = mask.to(self.args.local_rank)
                        prompt = prompt.to(self.args.local_rank)
                        batch_x_mark = batch_x_mark.to(self.args.local_rank)
                        batch_y = batch_y.to(self.args.local_rank)
                        y_mark = y_mark.to(self.args.local_rank)
                    if generate:
                        outputs = self.model.generate(
                            batch_x, batch_x_mark, prompt, mask, y_mark
                        )
                    else:
                        outputs = self.model(batch_x, batch_x_mark, prompt, mask)

                    f_dim = -1 if self.args.features == "MS" else 0
                    reco_outputs = outputs[:, : seq_len, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    reco_outputs = reco_outputs.detach().cpu().float().numpy()
                    batch_x = batch_x.detach().cpu().float().numpy()

                    reco.append(reco_outputs)
                    reco_trues.append(batch_x)

                    pred_outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                    pred_outputs = pred_outputs.detach().cpu().float().numpy()
                    batch_y = batch_y.detach().cpu().float().numpy()

                    preds.append(pred_outputs)
                    pred_trues.append(batch_y)

            preds = np.concatenate(preds, axis=0)
            pred_trues = np.concatenate(pred_trues, axis=0)

            reco = np.concatenate(reco, axis=0)
            reco_trues = np.concatenate(reco_trues, axis=0)

            pred_mae, pred_mse, pred_rmse, pred_mape, pred_mspe, pred_cr = metric(
                preds, pred_trues
            )
            reco_mae, reco_mse, reco_rmse, reco_mape, reco_mspe, reco_cr = metric(
                reco, reco_trues
            )
            logger.info(
                "[Reconstruct] domain:{}, mse:{}, mae:{}".format(
                    domain, reco_mse, reco_mae
                )
            )
            logger.info(
                "[Prection] domain:{}, mse:{}, mae:{}".format(
                    domain, pred_mse, pred_mae
                )
            )

            f = open(folder_path + domain + "_metrics.txt", "a")
            f.write(setting + "  \n")
            f.write(str(self.args))
            f.write("reconstruct metrics")
            f.write(
                "\nmse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}\n".format(
                    reco_mse, reco_mae, reco_rmse, reco_mape, reco_mspe
                )
            )
            f.write("prediction metrics")
            f.write(
                "\nmse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(
                    pred_mse, pred_mae, pred_rmse, pred_mape, pred_mspe
                )
            )
            f.write("\n")
            f.close()

            np.save(folder_path + domain + "_predicton.npy", preds)
            np.save(folder_path + domain + "_pred_true.npy", pred_trues)
            np.save(folder_path + domain + "_reconstruct.npy", reco)
            np.save(folder_path + domain + "_reco_true.npy", reco_trues)

            plot_all_result(
                preds,
                pred_trues,
                folder_path + "images/" + domain + "_prediction_all.png",
            )
            plot_all_result(
                reco,
                reco_trues,
                folder_path + "images/" + domain + "_reconstruct_all.png",
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
