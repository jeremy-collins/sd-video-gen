import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from transformer import Transformer
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
import argparse
import cv2
import os

class MovingMNIST(data.Dataset):
    def __init__(self, num_frames=20, stride=1, path='mnist_test_seq.npy', stage='raw', shuffle=True):
        self.stage = stage
        self.num_frames = num_frames
        self.stride = stride
        self.path = path
        # self.indices, self.dataset = self.get_data(shuffle=shuffle)
        self.dataset = np.load(path)
        self.dataset = np.transpose(self.dataset, (1,0, 2, 3))
        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.8)]
        self.test_dataset = self.dataset[int(len(self.dataset) * 0.8):]
        self.active_dataset = self.train_dataset if stage == 'train' else self.test_dataset
        if shuffle:
            np.random.shuffle(self.active_dataset)
        self.active_dataset = self.active_dataset[:, :num_frames*stride:stride, :, :]
        # convert to an image. 1 corresponds to (255, 255, 255) and 0 corresponds to (0, 0, 0)
        self.active_dataset = np.stack([self.active_dataset, self.active_dataset, self.active_dataset], axis=4)

    def __getitem__(self, index):
        indices = [f"{index:04d}{j:03d}" for j in range(0, self.num_frames * self.stride, self.stride)]
        assert len(indices) == self.num_frames
        return indices, self.active_dataset[index]

    def __len__(self):
        return len(self.active_dataset)

if __name__ == '__main__':
    # dataset = BouncingBall(num_frames=5, stride=1, dir='data/complex_large_10_22', stage='train', shuffle=True)
    dataset = MovingMNIST(num_frames=20, stride=1, path='mnist_test_seq.npy', stage='train', shuffle=True)

    
    for i in range(1):
        print('path: ', dataset.path)
        print('clip ', i)
        print("clips in the dataset: ", len(dataset.dataset))
        print('clip length: ', len(dataset[0]))
        datapoint = dataset[i]
        frames = datapoint[1]
        print('frame shape: ', frames.shape)
        
        # for frame in frames[1]:
        #     cv2.imshow('frame', np.array(frame))
        #     cv2.waitKey(0)