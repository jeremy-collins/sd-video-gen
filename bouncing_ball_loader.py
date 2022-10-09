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

class BouncingBall(data.Dataset):
    def __init__(self, num_frames=5, fps=30, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True):
        self.stage = stage
        self.dir = dir
        self.num_frames = num_frames
        self.fps = fps
        self.dataset = self.get_data(shuffle=shuffle)

    def __getitem__(self, index):
        # obtaining file paths
        frame_names = self.dataset[index]

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            # frame = self.transform(frame) # TODO: add transforms
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame = frame.float()/255.0
            
            frames.append(frame)

        return frames

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        self.dataset = []

        # crawling the directory
        for dir, _, files in os.walk(self.dir):
            for file in files:
                # saving (time, path)
                parent = dir.split('/')[-1]

                # (parent+index, name)
                if file.endswith('.png'):
                    img_names.append((float(parent+file[-7:-4]), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        img_names = sorted(img_names, key=lambda x: x[0])

        # self.indices = [x[0] for x in img_names]

        for i in range(0, len(img_names), self.num_frames):
            frame_names = []
            for j in range(0, self.num_frames, 30 // self.fps):
                if i+j < len(img_names):
                    frame_names.append(img_names[i+j][1])
                else:
                    continue
                
            # each element is a list of frame names with length num_frames and skipping frames according to fps    
            self.dataset.append((frame_names))

        if shuffle:
            np.random.shuffle(self.dataset)
        else:
            self.dataset = np.array(self.dataset)

        return self.dataset

if __name__ == '__main__':
    dataset = BouncingBall(num_frames=5, fps=30, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True)
    
    for i in range(10):
        print('clip ', i)
        print("clips in the dataset: ", len(dataset.dataset))
        print('clip length: ', len(dataset[0]))
        print('frame shape: ', dataset[0][0].shape)
        frames = dataset[i]
        for frame in frames:
            print(frame.size())
            frame = frame.permute(1, 2, 0)
            cv2.imshow('frame', np.array(frame))
            cv2.waitKey(0)