from layers.Prompts import TimePrompt
from torch.utils.data import DataLoader, DistributedSampler
import warnings
from typing import List
import os
from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
)
from einops import rearrange
from copy import deepcopy

warnings.filterwarnings("ignore")


class MixDataLoader:
    def __init__(self, flag: str, config, tokenizer) -> None:
        self.config = config
        self.distribute = os.path.isfile(self.config.deepspeed_config)
        self.flag = flag
        dataset_dict = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "Weather": Dataset_Custom,
            "Electricity": Dataset_Custom,
            "Exchange": Dataset_Custom,
            "Illness": Dataset_Custom,
            "Traffic": Dataset_Custom,
        }
        self.time_prompt = TimePrompt(
            self.config.data_description_path,
            tokenizer=tokenizer,
            ts_token_num=self.config.q_num,
        )
        self.domains = deepcopy(self.config.domains)
        temp_dict = {}
        pop_list = []
        for d in self.domains.keys():
            channel_split = self.domains[d].get("channel_split")
            steps = self.domains[d].get("steps", 1)
            if channel_split:
                for i in range(0, len(self.time_prompt.channels[d]), channel_split):
                    temp_dict.update({
                        "{}-split{}".format(d, i//channel_split):{
                            "data_path": self.domains[d]["data_path"],
                            "target": self.domains[d]["target"] if i+channel_split > len(self.time_prompt.channels[d]) else str(i+channel_split-1),
                            "freq": self.domains[d]["freq"],
                            "kwargs": {
                                "channels": [i, min(i+channel_split, len(self.time_prompt.channels[d]))],
                                "steps": steps
                            }
                        }
                    })
                pop_list.append(d)
            elif steps:
                self.domains[d]["kwargs"] = {"steps": steps}
        for d in pop_list:
            self.domains.pop(d)
        self.domains.update(temp_dict)
        self.domain_list: List[str] = list(self.domains.keys())
        data = {
            k: dataset_dict[k.split("-")[0]](
                root_path=self.config.data_dir,
                flag=flag,
                size=[self.config.seq_len, self.config.label_len, self.config.pred_len],
                data_path=self.domains[k]["data_path"],
                features=self.config.features,
                target=self.domains[k]["target"],
                scale=self.config.scale,
                percent=self.config.percent,
                freq=self.domains[k]["freq"],
                split=self.config.split,
                **self.domains[k].get("kwargs", {})
            )
            for k in self.domain_list
        }
        if flag == "train":
            self.batch_size = self.config.batch_size
        else:
            self.batch_size = getattr(
                self.config, "eval_batch_size", self.config.batch_size
            )
        self.samplers = {
            k: DistributedSampler(data[k]) if self.distribute else None
            for k in self.domain_list
        }
        self.loaders = {
            k: DataLoader(
                dataset=data[k],
                batch_size=self.batch_size,
                shuffle=None if self.distribute else flag == "train",
                num_workers=self.config.num_workers,
                drop_last=True if flag == "train" else False,
                sampler=self.samplers[k],
            )
            for k in self.domain_list
        }
        self.batch_nums = [len(loader) for loader in self.loaders.values()]
        self.__length = sum(self.batch_nums)
        self.reset()

    def __len__(self):
        return self.__length

    def get_batch(self, domain: int) -> tuple:

        domain_str = self.domain_list[domain]
        channels = self.domains[domain_str].get("kwargs", {}).get("channels", None)
        x, y, x_mark, y_mark, timestamps = next(self.iters[domain_str])
        timestamps = None
        prompt = self.time_prompt.get_prompt(domain_str.split("_")[0].split("-")[0], timestamps=timestamps, channels=channels)

        return (
            prompt["input_ids"].repeat(x.shape[0], 1, 1),
            prompt["attention_mask"].repeat(x.shape[0], 1, 1),
            x,
            y,
            x_mark,
            y_mark,
        )

    def reset(self, domain=None, epoch=0):
        if domain is None:
            self.iters = {k: v._get_iterator() for k, v in self.loaders.items()}
            if self.distribute:
                for k in self.domain_list:
                    self.samplers[k].set_epoch(epoch)
        else:
            domain = self.domain_list[domain]
            self.iters[domain] = self.loaders[domain]._get_iterator()
            if self.distribute:
                self.samplers[domain].set_epoch(epoch)


class PreTrainDataloader(MixDataLoader):
    # Used during pre-training, with unfixed input length and fixed output.

    def __init__(self, flag: str, config, tokenizer) -> None:
        self.config = config
        self.seq_lens = config.pretrain_seq_lens
        assert isinstance(self.seq_lens, list), "Input length must be a list"

        self.distribute = os.path.isfile(self.config.deepspeed_config)
        self.flag = flag
        dataset_dict = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "Weather": Dataset_Custom,
            "Electricity": Dataset_Custom,
            "Exchange": Dataset_Custom,
            "Illness": Dataset_Custom,
            "Traffic": Dataset_Custom,
        }
        self.domains = deepcopy(self.config.domains)
        self.time_prompt = TimePrompt(
            self.config.data_description_path,
            tokenizer=tokenizer,
            ts_token_num=self.config.q_num,
        )
        temp_dict = {}
        pop_list = []
        for d in self.domains.keys():
            channel_split = self.domains[d].get("channel_split")
            steps = self.domains[d].get("steps", 1)
            if channel_split:
                for i in range(0, len(self.time_prompt.channels[d]), channel_split):
                    temp_dict.update({
                        "{}-split{}".format(d, i//channel_split):{
                            "data_path": self.domains[d]["data_path"],
                            "target": self.domains[d]["target"] if i+channel_split > len(self.time_prompt.channels[d]) else str(i+channel_split-1),
                            "freq": self.domains[d]["freq"],
                            "kwargs": {
                                "channels": [i, min(i+channel_split, len(self.time_prompt.channels[d]))],
                                "steps": steps
                            }
                        }
                    })
                pop_list.append(d)
            elif steps:
                self.domains[d]["kwargs"] = {"steps": steps}
        for d in pop_list:
            self.domains.pop(d)
        self.domains.update(temp_dict)
        data = {
            "{}_{}".format(d, s): dataset_dict[d.split("-")[0]](
                root_path=self.config.data_dir,
                flag=flag,
                size=[
                    s,
                    self.config.label_len,
                    self.config.single_pred_len,
                ],
                data_path=self.domains[d]["data_path"],
                features=self.config.features,
                target=self.domains[d]["target"],
                scale=self.config.scale,
                split=self.config.split,
                freq=self.domains[d]["freq"],
                **self.domains[d].get("kwargs", {})
            )
            for d in self.domains.keys()
            for s in self.seq_lens
        }
        self.domain_list: List[str] = list(data.keys())

        if flag == "train":
            self.batch_size = self.config.batch_size
        else:
            self.batch_size = getattr(
                self.config, "eval_batch_size", self.config.batch_size
            )

        self.samplers = {
            k: DistributedSampler(data[k]) if self.distribute else None
            for k in self.domain_list
        }
        self.loaders = {
            k: DataLoader(
                dataset=data[k],
                batch_size=self.batch_size,
                shuffle=None if self.distribute else flag == "train",
                num_workers=self.config.num_workers,
                drop_last=True if flag == "train" else False,
                sampler=self.samplers[k],
            )
            for k in self.domain_list
        }
        self.batch_nums = [len(loader) for loader in self.loaders.values()]
        self.__length = sum(self.batch_nums)

        self.reset()

    def __len__(self):
        return self.__length

    def get_batch(self, domain: int) -> tuple:
        domain_str = self.domain_list[domain]
        channels = self.domains[domain_str.split("_")[0]].get("kwargs", {}).get("channels")
        x, y, x_mark, y_mark, timestamps = next(self.iters[domain_str])
        # timestamps = None
        prompt = self.time_prompt.get_prompt(domain_str.split("_")[0].split("-")[0], timestamps=timestamps, channels=channels)

        return (
            rearrange(prompt["input_ids"], "(b c) n -> b c n", b=x.shape[0]) if timestamps else prompt["input_ids"].repeat(x.shape[0], 1, 1),
            rearrange(prompt["attention_mask"], "(b c) n -> b c n", b=x.shape[0]) if timestamps else prompt["attention_mask"].repeat(x.shape[0], 1, 1),
            x,
            y,
            x_mark,
            y_mark,
        )

    def reset(self, domain=None, epoch=0):
        if domain is None:
            self.iters = {k: v._get_iterator() for k, v in self.loaders.items()}
            if self.distribute:
                for k in self.domain_list:
                    self.samplers[k].set_epoch(epoch)
        else:
            domain = self.domain_list[domain]
            self.iters[domain] = self.loaders[domain]._get_iterator()
            if self.distribute:
                self.samplers[domain].set_epoch(epoch)


class EvalDataloader(MixDataLoader):
    # Used during evaluation, with fixed input length and unfixed output.

    def __init__(self, flag: str, config, tokenizer) -> None:
        self.config = config
        # Output length is a list
        self.pred_lens = config.tasks_lens
        assert isinstance(self.pred_lens, list), "Output length must be a list"

        self.distribute = os.path.isfile(self.config.deepspeed_config)
        self.flag = flag
        dataset_dict = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "Weather": Dataset_Custom,
            "Electricity": Dataset_Custom,
            "Exchange": Dataset_Custom,
            "Illness": Dataset_Custom,
            "Traffic": Dataset_Custom,
            "Electricity-eval": Dataset_Custom,
            "Weather-eval": Dataset_Custom,
            "Traffic-eval": Dataset_Custom,
        }
        self.time_prompt = TimePrompt(
            self.config.data_description_path,
            tokenizer=tokenizer,
            ts_token_num=self.config.q_num,
        )
        self.domains = deepcopy(self.config.domains)
        temp_dict = {}
        pop_list = []
        for d in self.domains.keys():
            channel_split = self.domains[d].get("channel_split")
            steps = self.domains[d].get("steps")
            if channel_split:
                for i in range(0, len(self.time_prompt.channels[d]), channel_split):
                    temp_dict.update({
                        "{}-split{}".format(d, i//channel_split):{
                            "data_path": self.domains[d]["data_path"],
                            "target": self.domains[d]["target"] if i+channel_split > len(self.time_prompt.channels[d]) else str(i+channel_split-1),
                            "freq": self.domains[d]["freq"],
                            "kwargs": {
                                "channels": [i, min(i+channel_split, len(self.time_prompt.channels[d]))]
                            }
                        }
                    })
                pop_list.append(d)
            elif steps:
                self.domains[d]["kwargs"] = {"steps": steps}
        for d in pop_list:
            self.domains.pop(d)
        self.domains.update(temp_dict)
        data = {
            "{}_{}".format(d, s): dataset_dict[d.split("-")[0]](
                root_path=self.config.data_dir,
                flag=flag,
                size=[
                    self.config.seq_len,
                    self.config.label_len,
                    s,
                ],
                data_path=self.domains[d]["data_path"],
                features=self.config.features,
                target=self.domains[d]["target"],
                scale=self.config.scale,
                percent=self.config.percent,
                freq=self.domains[d]["freq"],
                **self.domains[d].get("kwargs", {})
            )
            for d in self.domains.keys()
            for s in self.pred_lens
        }
        self.domain_list: List[str] = list(data.keys())

        if flag == "train":
            self.batch_size = self.config.batch_size
        else:
            self.batch_size = getattr(
                self.config, "eval_batch_size", self.config.batch_size
            )
        self.samplers = {
            k: DistributedSampler(data[k]) if self.distribute else None
            for k in self.domain_list
        }
        self.loaders = {
            k: DataLoader(
                dataset=data[k],
                batch_size=self.batch_size,
                shuffle=None if self.distribute else flag == "train",
                num_workers=self.config.num_workers,
                drop_last=True if flag == "train" else False,
                sampler=self.samplers[k],
            )
            for k in self.domain_list
        }
        self.batch_nums = [len(loader) for loader in self.loaders.values()]
        self.__length = sum(self.batch_nums)
        self.reset()

    def __len__(self):
        return self.__length

    def get_batch(self, domain: int) -> tuple:
        domain_str = self.domain_list[domain]
        channels = self.domains[domain_str.split("_")[0]].get("kwargs", {}).get("channels")

        x, y, x_mark, y_mark, timestamps = next(self.iters[domain_str])
        prompt = self.time_prompt.get_prompt(domain_str.split("_")[0].split("-")[0], timestamps=timestamps, channels=channels)
        return (
            rearrange(prompt["input_ids"], "(b c) n -> b c n", b=x.shape[0]) if timestamps else prompt["input_ids"].repeat(x.shape[0], 1, 1),
            rearrange(prompt["attention_mask"], "(b c) n -> b c n", b=x.shape[0]) if timestamps else prompt["attention_mask"].repeat(x.shape[0], 1, 1),
            x,
            y,
            x_mark,
            y_mark,
        )

    def reset(self, domain=None, epoch=0):
        if domain is None:
            self.iters = {k: v._get_iterator() for k, v in self.loaders.items()}
            if self.distribute:
                for k in self.domain_list:
                    self.samplers[k].set_epoch(epoch)
        else:
            domain = self.domain_list[domain]
            self.iters[domain] = self.loaders[domain]._get_iterator()
            if self.distribute:
                self.samplers[domain].set_epoch(epoch)

