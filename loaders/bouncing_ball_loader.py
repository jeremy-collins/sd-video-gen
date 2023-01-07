import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from models.transformer import Transformer
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
import argparse
import cv2
import os

class BouncingBall(data.Dataset):
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.indices, self.dataset = self.get_data(shuffle=shuffle)

    def __getitem__(self, index):
        # obtaining file paths
        frame_names = self.dataset[index]

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            # frame = self.transform(frame) # TODO: add transforms
            frames.append(frame)
        
        frames = np.stack(frames, axis=0)
                    
        #  frames.shape: (seq_len, dim_model)   
        return self.indices[index], frames

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        dataset = []
        indices = []

        # crawling the directory
        for dir, _, files in os.walk(self.dir):
            for file in files:
                # saving (time, path)
                parent = dir.split('/')[-1]
                # (parent+index, name)
                if file.endswith('.png'):
                    img_names.append((int(parent+file[-7:-4]), os.path.join(dir, file)))

        # sorting the names numerically. first 4 digits are folder and last 3 are file
        img_names = sorted(img_names, key=lambda x: x[0])

        # indices = [x[0] for x in img_names]

        for i in range(0, len(img_names) - self.num_frames * self.stride + 1, self.num_frames * self.stride):
            index_list = []
            frame_names = []
            for j in range(self.stride): # don't miss the skipped frames from the stride
                if i % self.stride == j:
                    for k in range(self.num_frames): # for each sequence
                        correct_parent = img_names[i][1].split('/')[-2] # all frames in the sequence should have the same parent folder
                        current_parent = img_names[i+k*self.stride][1].split('/')[-2]
                        if correct_parent == current_parent:
                            index_list.append(img_names[i+k*self.stride][0]) # getting frame i, i+self.stride, i+2*self.stride, ... (i+1)+self.stride, (i+1)+2*self.stride, ... etc
                            frame_names.append(img_names[i+k*self.stride][1])
                        else:
                            break # parent folder mismatch, so don't add this sequence

                    # list of lists of frame indices
                    indices.append(index_list)

                    # each element is a list of frame names with length num_frames and skipping frames according to stride
                    dataset.append(frame_names)
            
            # # list of lists of frame indices 
            # indices.append(index_list) 
                
            # # each element is a list of frame names with length num_frames and skipping frames according to stride    
            # dataset.append(frame_names)
            
        if shuffle:
            np.random.shuffle(dataset)
        else:
            dataset = np.array(dataset)

        return indices, dataset

if __name__ == '__main__':
    # dataset = BouncingBall(num_frames=5, stride=1, dir='data/complex_large_10_22', stage='train', shuffle=True)
    dataset = BouncingBall(num_frames=5, stride=3, dir='data/simple_bw_3000', stage='train', shuffle=True)

    
    for i in range(10):
        print('dir: ', dataset.dir)
        print('clip ', i)
        print("clips in the dataset: ", len(dataset.dataset))
        print('clip length: ', len(dataset[0]))
        datapoint = dataset[i]
        frames = datapoint[1]
        print('frame shape: ', frames.shape) # (5, 64, 64, 3)
        for frame in frames:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)