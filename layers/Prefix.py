from transformers import AutoTokenizer
import json

class DatasetDescroption:
    def __init__(self, json_file_path="configs/data_description.json") -> None:
        """
        ETT: https://github.com/zhouhaoyi/ETDataset
        Weather: https://www.bgc-jena.mpg.de/wetter/
        Weather: https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf
        Electricity: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.datasets = data.get("datasets", {})
        self.target = data.get("target", {})
        self.exogenous_variable = data.get("exogenous_variable", {})

    def merge_channels(self):
        merged_dict = {}
        for key in self.target:
            merged_dict[key] = self.exogenous_variable.get(key, []) + [self.target[key]]
        return merged_dict
