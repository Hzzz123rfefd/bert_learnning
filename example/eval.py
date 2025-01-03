import argparse
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.getcwd())
from rerankai import datasets, models
from rerankai.utils import *

def main(args):
    config = load_config(args.model_config_path)
    """ get net struction """
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])

    """ get data loader """
    eval_datasets = datasets[config["dataset_type"]](**config["dataset"], tokenizer = net.tokenizer, data_type = "train")
    eval_dataloader = DataLoader(
        eval_datasets, 
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = eval_datasets.collate_fn
    )
    
    """ eval """
    log_message, accuracy, precision, recall, f1 = net.eval_model(epoch = 0, val_dataloader = eval_dataloader, log_path = None)
    print(log_message)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/rerank.yml")
    args = parser.parse_args()
    main(args)