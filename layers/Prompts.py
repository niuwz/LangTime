from transformers import AutoTokenizer
from layers.Prefix import DatasetDescroption

class TimePrompt:
    def __init__(self, data_description_path, tokenizer, ts_token_num=1, patch_size=8) -> None:
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.ts_token_num = ts_token_num
        data_descroption = DatasetDescroption(data_description_path)
        self.datasets = data_descroption.datasets
        self.channels = data_descroption.merge_channels()


    def get_prompt(self, domain:str, timestamps=None, channels=None):
        infos = []
        if channels:
            domain_channels = self.channels[domain][channels[0]:channels[1]]
        else:
            domain_channels = self.channels[domain]
        if timestamps:
            for timestamp in timestamps:
                for channel in domain_channels:
                    info = f"dataset:{self.datasets[domain]}, channel:{channel}, period:{timestamp}, value:{'<|TS_ENC|>' * self.ts_token_num}"
                    infos.append(info)
        else:
            for channel in domain_channels:
                info = f"dataset:{self.datasets[domain]}, channel:{channel}, value:{'<|TS_ENC|>' * self.ts_token_num}"
                infos.append(info)
        content = []
        for info in infos:
            content.append(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, and your target is to summarize a time series and predict the next time series base on the given information. "
                    },
                    {
                        "role": "user",
                        "content": "The information of given time series: {}, compress this series into one word: <|ts_emb|>. \nBase on the given information, predict the next value:".format(
                            info
                        ),
                    },
                    {"role": "assistant", "content": "<|ts_out|>"},
                ]
            )
        prompt = self.tokenizer.apply_chat_template(
            content, return_tensors="pt", padding=True, return_dict=True)
        return prompt