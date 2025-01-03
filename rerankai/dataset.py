import math
import random
from typing import List
import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from rerankai.utils import *


class DatasetForRerank(Dataset):
    def __init__(
            self,
            train_data_path:str = None,
            test_data_path:str = None,
            valid_data_path:str = None,
            max_padding_length: int = 512,
            tokenizer: PreTrainedTokenizer = None,
            train_group_size = 8,
            data_type:str = "train"
    ):
        # load data
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path,split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path,split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path,split = "train")

        self.train_group_size = train_group_size
        self.total_len = len(self.dataset)
        self.max_padding_length = max_padding_length
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        query = self.dataset[idx]['query']
        pos = random.choice(self.dataset[idx]['pos'])
        if len(self.dataset[idx]['neg']) < self.train_group_size - 1:
            num = math.ceil((self.train_group_size - 1) / len(self.dataset[idx]['neg']))
            negs = random.sample(self.dataset[idx]['neg'] * num, self.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[idx]['neg'], self.train_group_size - 1)

        querys = [query for i in range(self.train_group_size)]
        docs = [pos] + negs
        output = self.create_one_example(querys, docs)
        output["label"] = torch.tensor([0], dtype=torch.long)
        return output

    def create_one_example(self, querys: List[str], docs: List[str]):
        item = self.tokenizer(
            querys,
            docs,
            truncation = True,
            max_length = 512,
            padding = True,
            return_tensors = "pt"
        )
        return item

    def collate_fn(self, batch):
        max_length = max(item["input_ids"].shape[1] for item in batch)

        for item in batch:
            seq_length = item['input_ids'].shape[1]
            input_ids = torch.ones((item['input_ids'].shape[0], max_length), dtype = item['input_ids'].dtype) * self.tokenizer.pad_token_id
            attention_mask = torch.zeros((item['attention_mask'].shape[0], max_length), dtype = item['attention_mask'].dtype)
            if "token_type_ids" in item:
                token_type_ids = torch.zeros((item['token_type_ids'].shape[0], max_length), dtype = item['token_type_ids'].dtype)
                token_type_ids[:, :seq_length] = item['token_type_ids']
                item['token_type_ids'] = token_type_ids

            input_ids[:, :seq_length] = item['input_ids']
            attention_mask[:, :seq_length] = item['attention_mask']

            item["input_ids"] = input_ids
            item["attention_mask"] = attention_mask

        return  recursive_collate_fn(batch)