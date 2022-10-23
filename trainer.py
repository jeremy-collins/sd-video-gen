import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import math
import numpy as np
from transformer import Transformer
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from bouncing_ball_loader import BouncingBall
import argparse
import os
from tqdm import tqdm
# from sd_utils import SDUtils
import cv2

import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from base64 import b64encode

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import os

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        # self.sd_utils = SDUtils()
        model = Transformer()
        self.SOS_token = torch.ones((1, model.dim_model), dtype=torch.float32, device=self.device) * 2


        auth_token = os.environ['HF_TOKEN']
        print('loading VAE...')
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
        self.vae = self.vae.to(self.device)
    
    def encode_img(self, imgs):
        # turn an image into image latents
        if not isinstance(imgs, list):
            imgs = [imgs]

        img_arr = np.stack([np.array(img) for img in imgs], axis=0)
        img_arr = img_arr / 255.0
        img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = self.vae.encode(img_arr.to(self.device))
        # latent_dists = self.vae.encode(img_arr)
        latent_samples = latent_dists.sample()
        latent_samples *= 0.18215

        return latent_samples  

    def train_loop(self, model, opt, loss_fn, dataloader, frames_to_predict):
        model = model.to(self.device)
        model.train()
        total_loss = 0
        # sd_utils = SDUtils()
            
        for i, (index_list, batch) in enumerate(tqdm(dataloader)):
            # if i >= 199:
            #     break
            
            # X, y = batch[:, 0], batch[:, 1]
            
            # X = batch[:, :-1]
            # y = batch[:,-1].unsqueeze(1)

            X = batch
            y = batch
            
            X = torch.tensor(X).to(self.device)
            y = torch.tensor(y).to(self.device)
            
            # y_input = y
            # y_expected = y
            
            # shift the tgt by one so we always predict the next embedding
            # y_input = y[:,:-1] # all but last 
            y_input = y # because we don't have an EOS token
            y_expected = y[:,1:] # all but first because the prediction is shifted by one
            
            y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
            y_expected = y_expected.permute(1, 0, 2)
            
            # Get mask to mask out the future frames
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
        
            # X shape is (batch_size, src sequence length, input.shape)
            # y_input shape is (batch_size, tgt sequence length, input.shape)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(X, y_input, tgt_mask)
            # pred = None
            
            # Permute pred to have batch size first again
            # pred = pred.permute(1, 2, 0)
            
            # check decoding
            # gt = y_expected[-1][-1].reshape((1, 4, 8, 8))
            # gt_reconstruction = sd_utils.decode_img_latents(gt)
            # gt_reconstruction = np.array(gt_reconstruction[0])
            # cv2.imshow('gt_reconstruction', gt_reconstruction)
            # cv2.waitKey(0)
            
            loss = loss_fn(pred[-1], y_expected[-1])

            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)
        # return total_loss / 200.0

    def validation_loop(self, model, loss_fn, dataloader, frames_to_predict):  
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for j, (index_list, batch) in enumerate(tqdm(dataloader)):
                # if j >= 49:
                #     break
            
                # X, y = batch[:, 0], batch[:, 1]
                
                # X = batch[:, :-1]
                # y = batch[:,-1].unsqueeze(1)
                
                X = batch
                y = batch
                
                X = torch.tensor(X).to(self.device)
                y = torch.tensor(y).to(self.device)
                
                y_input = y
                y_expected = y
                
                y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
                y_expected = y_expected.permute(1, 0, 2)
                
                # Get mask to mask out the next words
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
            
                # X shape is (batch_size, src sequence length, input.shape)
                # y_input shape is (batch_size, tgt sequence length, input.shape)

                # Standard training except we pass in y_input and tgt_mask
                pred = model(X, y_input, tgt_mask)
                # pred = None

                # Permute pred to have batch size first again
                # pred = pred.permute(1, 2, 0)
                
                loss = loss_fn(pred[-1], y_expected[-1])
                total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)
        # return total_loss / 50.0

    def fit(self, model, opt, loss_fn, train_dataloader, val_dataloader, epochs, frames_to_predict): 
        # Used for plotting later on
        train_loss_list, validation_loss_list = [], []
        
        print("Training and validating model")
        for epoch in range(epochs):
            if epochs > 1:
                print("-"*25, f"Epoch {epoch + 1}","-"*25)
            
            train_loss = self.train_loop(model, opt, loss_fn, train_dataloader, frames_to_predict)
            train_loss_list += [train_loss]
            
            validation_loss = self.validation_loop(model, loss_fn, val_dataloader, frames_to_predict)
            validation_loss_list += [validation_loss]
            
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
        
        # counting number of files in ./checkpoints
        index = len(os.listdir('./checkpoints'))    
        
        if epochs > 1:
            # save model
            torch.save(model.state_dict(), './checkpoints/model' + '_' + str(index) + '.pt')
            print('model saved as model' + '_' + str(index) + '.pt')
            
        return train_loss_list, validation_loss_list

    def collate_embeddings(self, batch):
        # turn list of images into a batch of embeddings
        new_batch = []
        for i, (index_list, frames) in enumerate(batch):
            new_frames = []
            # each element in batch is a tuple of (List(index_list), List(np.array(frames))
            for frame in frames:
                frame = self.encode_img(frame)
                frame = frame.squeeze(0)
                frame = frame.flatten()
                new_frames.append(frame)

            new_frames = torch.stack(new_frames, dim=0)
            # frames = frames.detach()
            # frames.requires_grad = False

            # concatenating SOS token, 
            new_frames = torch.cat((self.SOS_token, new_frames), dim=0)
            new_batch.append((index_list, new_frames))
            
        return torch.utils.data.dataloader.default_collate(new_batch)

    def custom_collate(self, batch):
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_best', type=bool, default=False)
    args = parser.parse_args()
    
    # torch.multiprocessing.set_start_method('spawn')
    
    frames_per_clip = 5
    frames_to_predict = 1
    stride = 1 # number of frames to shift when loading clips
    batch_size = 32
    epoch_ratio = 1 # to sample just a portion of the dataset
    epochs = 10
    lr = 0.00001
    num_workers = 0

    dim_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout_p = 0.1

    trainer = Trainer()
    
    model = Transformer(num_tokens=0, dim_model=dim_model, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout_p=dropout_p)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
    
    if args.dataset == 'ucf':    
        ucf_data_dir = "/Users/jsikka/Documents/UCF-101"
        ucf_label_dir = "/Users/jsikka/Documents/ucfTrainTestlist"
        

        tfs = transforms.Compose([
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                # rescale to the most common size
                transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
        ])

        train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                        step_between_clips=stride, train=True, transform=tfs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=trainer.custom_collate)
        # create test loader (allowing batches and other extras)
        test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                            step_between_clips=stride, train=False, transform=tfs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=trainer.custom_collate)
        
    elif args.dataset == 'ball':
        # train_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='train', shuffle=True)
        # train_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_blackwhite1/content/2D-bouncing/test3_bouncing_ball', stage='train', shuffle=True)
        # train_dataset = BouncingBall(num_frames=5, stride=stride, dir='/media/jer/data/tccvg/bouncing_ball_3000_blackwhite_simple1/content/2D-bouncing/test2_simple_bouncing_ball', stage='train', shuffle=True)
        train_dataset = BouncingBall(num_frames=5, stride=stride, dir='./data/bouncing_ball_3000_blackwhite_simple1/content/2D-bouncing/test2_simple_bouncing_ball', stage='train', shuffle=True)
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers)
        
        # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='test', shuffle=True)
        # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_blackwhite1/content/2D-bouncing/test3_bouncing_ball', stage='test', shuffle=True)
        test_dataset = BouncingBall(num_frames=5, stride=stride, dir='./data/bouncing_ball_3000_blackwhite_simple1/content/2D-bouncing/test2_simple_bouncing_ball', stage='test', shuffle=True)
        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * epoch_ratio))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers)
        
    # # print(train_loader)
    # print("TRAIN LOADER")
    # for i in train_loader:
    #     print(len(i))
    #     print(i.size())
    #     print(i)
    #     break

    # print("TEST LOADER")
    # # print(test_loader)
    # for i in test_loader:
    #     print(i.size())
    #     print(i)
    #     break
    
    # train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_loader, test_loader, 10)


    if args.save_best:
        best_loss = 1e10
        epoch = 1
        while True:
            print("-"*25, f"Epoch {epoch}","-"*25)
            train_loss_list, validation_loss_list = trainer.fit(model=model, opt=opt, loss_fn=loss_fn, train_dataloader=train_loader, val_dataloader=test_loader, epochs=1, frames_to_predict=frames_to_predict)
            if validation_loss_list[-1] < best_loss:
                best_loss = validation_loss_list[-1]
                torch.save(model.state_dict(), './checkpoints/model_best.pt')
                print('model saved as model_best.pt')
            epoch += 1
    else:
        trainer.fit(model=model, opt=opt, loss_fn=loss_fn, train_dataloader=train_loader, val_dataloader=test_loader, epochs=epochs, frames_to_predict=frames_to_predict)