from sd_utils import SDUtils
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
    def __init__(self, num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='raw', shuffle=True):
        self.stage = stage
        self.dir = os.path.join(dir, stage)
        self.num_frames = num_frames
        self.stride = stride
        self.sd_utils = SDUtils()
        self.indices, self.dataset = self.get_data(shuffle=shuffle)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Transformer()       
        self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2
        # self.EOS_token = torch.ones((1, model.dim_model), dtype=torch.float32) * 3

    def __getitem__(self, index):
        # obtaining file paths
        frame_names = self.dataset[index]

        # loading and formatting image
        frames=[]
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            # frame = self.transform(frame) # TODO: add transforms

            frame = self.sd_utils.encode_img(frame)
           
        #     # check decoding
        #     # reconstruction = self.sd_utils.decode_img_latents(frame)
        #     # reconstruction = np.array(reconstruction[0])
        #     # cv2.imshow('reconstruction', reconstruction)
        #     # cv2.waitKey(0)
            
            frame = frame.squeeze(0)
        #     # frame = torch.from_numpy(frame)
        #     # frame = frame.permute(2, 0, 1)
        #     # frame = frame.float()/255.0
            
        #     # bs, seq_len, _, _, _ = frame.shape
            frame = frame.flatten()
                
            frames.append(frame)
        
        # # creating tensor with requires_grad=False
        frames = torch.stack(frames, dim=0)
        frames = frames.detach()
        frames.requires_grad = False
        
        # # concatenating SOS token, 
        frames = torch.cat((self.SOS_token, frames), dim=0)
                    
        #  frames.shape: (seq_len + 1, dim_model)   
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

        for i in range(0, len(img_names), self.num_frames):
            index_list = []
            frame_names = []
            for j in range(0, self.num_frames * self.stride, self.stride):
                if i+j < len(img_names):
                    index_list.append(img_names[i+j][0])
                    frame_names.append(img_names[i+j][1])
                else:
                    break
            
            # list of lists of frame indices 
            indices.append(index_list) 
                
            # each element is a list of frame names with length num_frames and skipping frames according to stride    
            dataset.append(frame_names)
            

        if shuffle:
            np.random.shuffle(dataset)
        else:
            dataset = np.array(dataset)
        
        return indices, dataset
        

if __name__ == '__main__':
    dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='train', shuffle=True)
    
    for i in range(10):
        print('dir: ', dataset.dir)
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