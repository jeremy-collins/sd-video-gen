import torch
import argparse
import os
import yaml
import argparse
from types import SimpleNamespace

def load_config(config_name):
    # config_name = remove_config_index(config_name)
    config_path = os.path.join('./config', config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    # data_obj = namedtuple('MyTuple', data)
    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj

def remove_config_index(config_name):
    # remove last string before _ from config name if it is a number
    if config_name[-1].isdigit():
        config_name = config_name[:config_name.rfind('_')]

    return config_name

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_best', type=bool, default=False)
    parser.add_argument('--folder', type=str, required=True) # data folder
    parser.add_argument('--config', type=str, required=True) # model/config name
    parser.add_argument('--resume', type=bool, default=False) # resume training from checkpoint
    parser.add_argument('--debug', type=bool, default=False) # turn off wandb logging

    parser.add_argument('--pred_frames', type=int, default=1) # number of frames to predict
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--fullscreen', type=bool, default=False)
    
    args = parser.parse_args()
    return load_config(args.config), args