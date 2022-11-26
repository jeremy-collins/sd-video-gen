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
from sd_utils import SDUtils
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
import wandb
import datetime
from config import parse_config_args
from transformers import get_linear_schedule_with_warmup
from contrastive_loss import BiPatchNCE

class Trainer():
    def __init__(self):
        self.config, self.args = parse_config_args()
        # counting number of files in ./checkpoints containing config name
        self.index = len([name for name in os.listdir('./checkpoints') if self.args.config in name])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.sd_utils = SDUtils()
        model = Transformer()
        # self.SOS_token = torch.ones((1, 1, model.dim_model), dtype=torch.float32, device=self.device) * 2
        self.SOS_token = torch.ones((1, 1, self.config.FRAME_SIZE ** 2 // 64 * 4), dtype=torch.float32, device=self.device) * 2
        # self.SOS_token = torch.ones((1, 4, 8, 8), dtype=torch.float32, device=self.device) * 2
        # auth_token = os.environ['HF_TOKEN']
    
    def check_decoding(self, latent, label='img', fullscreen=False):
        print('latent shape: ', latent.shape)
        latent = latent.reshape((1, 4, self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8)) # reshaping to SD latent shape
        # latent = latent.reshape((1, 4, 16, 16)) # reshaping to SD latent shape
        reconstruction = self.sd_utils.decode_img_latents(latent)
        reconstruction = np.array(reconstruction[0])
        if fullscreen:
            cv2.namedWindow(label, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(label, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(label, reconstruction)
        cv2.waitKey(0)

    def gradient_difference_loss(self, frameX_flattened, frameY_flattened, alpha=1):
        vert_hori_dim = int(np.sqrt(frameX_flattened.shape[-1] // 4))
        # vert_hori_dim = int(np.sqrt(frameX_flattened.shape[1]/4))
        # (5, 8, 1024) -> (5, 8, 4, 16, 16)
        frameX = torch.reshape(frameX_flattened, (frameX_flattened.shape[0], frameX_flattened.shape[1], 4, vert_hori_dim,vert_hori_dim))
        frameY = torch.reshape(frameY_flattened, (frameY_flattened.shape[0], frameX_flattened.shape[1], 4, vert_hori_dim,vert_hori_dim))
        vertical_gradient_X = frameX[:, :, :, 1:, :] - frameX[:, :, :, :-1, :]
        vertical_gradient_Y = frameY[:, :, :, 1:, :] - frameY[:, :, :, :-1, :]
        vertical_gradient_loss = torch.abs(torch.abs(vertical_gradient_X) - torch.abs(vertical_gradient_Y))

        horizontal_gradient_X = frameX[:, :, :, :, 1:] - frameX[:, :, :, :, :-1]
        horizontal_gradient_Y = frameY[:, :, :, :, 1:] - frameY[:, :, :, :, :-1]
        horizontal_gradient_loss = torch.abs(torch.abs(horizontal_gradient_X) - torch.abs(horizontal_gradient_Y))

        gdloss = torch.sum(torch.pow(vertical_gradient_loss, alpha)) + torch.sum(torch.pow(horizontal_gradient_loss, alpha))
        gdloss = gdloss / (frameX_flattened.numel()) # * 2) # normalizing
        return gdloss

    # def contrastive_loss(num_frames, batch_size, feat_height, feat_width, temperature):
    #     return BiPatchNCE(num_frames, batch_size, feat_height, feat_width, temperature)
    
    def criterion(self, use_mse=True, use_gdl=True, lambda_gdl=1, alpha=2, use_contrastive=True, temperature=0.07, lambda_contrastive=0.1):
        if use_mse and not use_gdl:
            return nn.MSELoss()
        elif use_gdl and not use_mse:
            return self.gradient_difference_loss
        elif use_mse and use_gdl and not use_contrastive:
            # return lambda x, y: nn.MSELoss()(x, y) + lambda_gdl * self.gradient_difference_loss(x[-1], y[-1], alpha)
            return lambda x, y: nn.MSELoss()(x, y) + lambda_gdl * self.gradient_difference_loss(x, y, alpha)
        elif use_mse and use_gdl and use_contrastive:
            # (tgt sequence length, batch size, input shape)
            frames_to_predict = self.config.FRAMES_TO_PREDICT[0] # TODO: make this work for multiple frames to predict in config
            batch_size = self.config.BATCH_SIZE[0] # TODO: make this work for multiple batch sizes in config
            feat_height = self.config.FRAME_SIZE // 8
            feat_width = self.config.FRAME_SIZE // 8
            # bpnce = BiPatchNCE(N=batch_size, T=frames_to_predict, h=feat_height, w=feat_width, temperature=temperature).to(self.device)
        #     return lambda x, y: nn.MSELoss()(x, y) \
        #             + lambda_gdl * self.gradient_difference_loss(x[-1], y[-1], alpha) \
        #             + lambda_contrastive * BiPatchNCE(N=int(x.numel() / (frames_to_predict * 4 * feat_height * feat_width)), T=frames_to_predict, h=feat_height, w=feat_width, temperature=temperature).to(self.device)(x.permute(1,0,2).reshape((-1, frames_to_predict, 4, feat_height, feat_width)), y.permute(1,0,2).reshape((-1, frames_to_predict, 4, feat_height, feat_width)))
            return lambda x, y: nn.MSELoss()(x, y) \
                    + lambda_gdl * self.gradient_difference_loss(x, y, alpha) \
                    + lambda_contrastive * BiPatchNCE(N=int(x.numel() / (frames_to_predict * 4 * feat_height * feat_width)), T=frames_to_predict, h=feat_height, w=feat_width, temperature=temperature).to(self.device)(x.permute(1,0,2).reshape((-1, frames_to_predict, 4, feat_height, feat_width)), y.permute(1,0,2).reshape((-1, frames_to_predict, 4, feat_height, feat_width)))
        else:
            print('Invalid loss function combination')
            return None

    def train_loop(self, model, opt, scheduler, loss_fn, dataloader, frames_to_predict):
        model = model.to(self.device)
        model.train()
        total_loss = 0
        mse_loss = 0
        gdl_loss = 0
        contrastive_loss = 0
            
        for i, (index_list, batch) in enumerate(tqdm(dataloader)):
            # print('batch shape: ', batch.shape)
            # turning batch of images into a batch of embeddings
            new_batch = self.sd_utils.encode_batch(batch, use_sos=True) # (batch size, frames_per_clip, 3, 128, 128) -> (batch size, frames_per_clip, 1024)
            new_batch = torch.tensor(new_batch).to(self.device)

            # shift the tgt by one so we always predict the next embedding
            y_input = new_batch[:,:-1] # all but last 

            # y_input = y # because we don't have an EOS token
            y_expected = new_batch[:,1:] # all but first because the prediction is shifted by one
            
            # y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1) # merging h w and c dims --> moved to encode_batch
            y_expected = y_expected.permute(1, 0, 2)
            
            # Get mask to mask out the future frames
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
        
            # X shape is (batch_size, src sequence length, input shape)
            # y_input shape is (batch_size, tgt sequence length, input shape)
            pred = model(new_batch, y_input, tgt_mask)
            new_batch_permuted = new_batch.permute(1, 0, 2)
            # output is (tgt sequence length, batch size, input shape)
            pred[-frames_to_predict:] = pred[-frames_to_predict:] + new_batch_permuted[-(frames_to_predict+1):-1]
            
            # loss = loss_fn(pred[-1], y_expected[-1])
            loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])
            # # print('mse: ', torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]))
            # wandb.log({'mse_train': torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:])})
            # # print('gdl: ', self.gradient_difference_loss(pred[-1], y_expected[-1]))
            # wandb.log({'gdl_train': self.gradient_difference_loss(pred[-1], y_expected[-1])})
            # # print('contrastive: ', loss - torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]) - self.gradient_difference_loss(pred[-1], y_expected[-1]))
            # wandb.log({'contrastive_train': loss - torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]) - self.gradient_difference_loss(pred[-1], y_expected[-1])})

            # contrastive_pred_input = pred[-frames_to_predict:].permute(1,0,2).reshape((self.config.BATCH_SIZE[0], frames_to_predict,4, self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8)).to(self.device)
            # contrastive_y_input = y_expected[-frames_to_predict:].permute(1,0,2).reshape((self.config.BATCH_SIZE[0], frames_to_predict,4, self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8)).to(self.device)
            # print('contrastive pred shape: ', contrastive_pred_input.shape)
            # print('contrastive y shape: ', contrastive_y_input.shape)
            # print('contrastive: ', BiPatchNCE(frames_to_predict, self.config.BATCH_SIZE[0], self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8, 0.07)(pred[-frames_to_predict:].permute(1,0,2).reshape((self.config.BATCH_SIZE[0], frames_to_predict,4, self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8)).to(self.device), y_expected[-frames_to_predict:].permute(1,0,2).reshape((self.config.BATCH_SIZE[0], frames_to_predict,4, self.config.FRAME_SIZE // 8, self.config.FRAME_SIZE // 8)).to(self.device)).to(self.device))

            # checking decoding
            # self.check_decoding(pred[0, -1], 'pred', fullscreen=True)
            # self.check_decoding(y_expected[0, -1], 'gt', fullscreen=True)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()

            mse = torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]).detach().item()
            # gdl = self.gradient_difference_loss(pred[-1], y_expected[-1]).detach().item()
            gdl = self.gradient_difference_loss(pred, y_expected).detach().item()
        
            mse_loss += mse
            gdl_loss += gdl
            contrastive_loss += loss.detach().item() - mse - gdl

            total_loss += loss.detach().item()

        train_loss = total_loss / len(dataloader)
        wandb.log({'train_loss': train_loss})

        wandb.log({'mse_train': mse_loss / len(dataloader)})
        wandb.log({'gdl_train': gdl_loss / len(dataloader)})
        wandb.log({'contrastive_train': contrastive_loss / len(dataloader)})
            
        return train_loss

    def validation_loop(self, model, loss_fn, dataloader, frames_to_predict):  
        model.eval()
        total_loss = 0
        mse_loss = 0
        gdl_loss = 0
        contrastive_loss = 0

        with torch.no_grad():
            for j, (index_list, batch) in enumerate(tqdm(dataloader)):
                # turning batch of images into a batch of embeddings
                new_batch = self.sd_utils.encode_batch(batch, use_sos=True)
                new_batch = torch.tensor(new_batch).to(self.device)

                # shift the tgt by one so we always predict the next embedding
                y_input = new_batch[:,:-1] # all but last 

                # y_input = y # because we don't have an EOS token
                y_expected = new_batch[:,1:] # all but first because the prediction is shifted by one
                
                # y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1) # merging h w and c dims --> moved to encode_batch
                y_expected = y_expected.permute(1, 0, 2)
                
                # Get mask to mask out the future frames
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(self.device)
            
                # X shape is (batch_size, src sequence length, input shape)
                # y_input shape is (batch_size, tgt sequence length, input shape)
                pred = model(new_batch, y_input, tgt_mask)
                # output is (tgt sequence length, batch size, input shape)
                new_batch_permuted = new_batch.permute(1, 0, 2)
                pred[-frames_to_predict:] = pred[-frames_to_predict:] + new_batch_permuted[-(frames_to_predict+1):-1]
                
                # loss = loss_fn(pred[-1], y_expected[-1])
                loss = loss_fn(pred[-frames_to_predict:], y_expected[-frames_to_predict:])

                #  # print('mse: ', torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]))
                # wandb.log({'mse_val': torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:])})
                # # print('gdl: ', self.gradient_difference_loss(pred[-1], y_expected[-1]))
                # wandb.log({'gdl_val': self.gradient_difference_loss(pred[-1], y_expected[-1])})
                # # print('contrastive: ', loss - torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]) - self.gradient_difference_loss(pred[-1], y_expected[-1]))
                # wandb.log({'contrastive_val': loss - torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]) - self.gradient_difference_loss(pred[-1], y_expected[-1])})

                # checking decoding
                # self.check_decoding(pred[0, -1], 'pred')
                # self.check_decoding(y_expected[0, -1], 'gt')

                mse = torch.nn.MSELoss()(pred[-frames_to_predict:], y_expected[-frames_to_predict:]).detach().item()
                gdl = self.gradient_difference_loss(pred, y_expected).detach().item()
            
                mse_loss += mse
                gdl_loss += gdl
                contrastive_loss += loss.detach().item() - mse - gdl

                total_loss += loss.detach().item()

        val_loss = total_loss / len(dataloader)
        wandb.log({'val_loss': val_loss})
        
        wandb.log({'mse_val': mse_loss / len(dataloader)})
        wandb.log({'gdl_val': gdl_loss / len(dataloader)})
        wandb.log({'contrastive_val': contrastive_loss / len(dataloader)})
            
        return val_loss

    def fit(self, model, opt, scheduler, loss_fn, train_dataloader, val_dataloader, frames_to_predict): 
        print("Training and validating model")
            
        train_loss = self.train_loop(model, opt, scheduler, loss_fn, train_dataloader, frames_to_predict)
        validation_loss = self.validation_loop(model, loss_fn, val_dataloader, frames_to_predict)
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

        return train_loss, validation_loss

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
            # filtered_batch.append((video, label))
            filtered_batch.append((label, video))
        return torch.utils.data.dataloader.default_collate(filtered_batch)

    
def main():
    config, args = parse_config_args()

    # appending current date to name
    # args.config = args.config + '_' + str(datetime.date.today())
    
    if args.debug:
        os.environ['WANDB_SILENT']="true"
        wandb.init(mode="disabled")
    else:
        wandb.init(config=wandb.config)

    # torch.multiprocessing.set_start_method('spawn')
    
    # frames_per_clip = 5
    # frames_to_predict = 5
    # stride = 1 # number of frames to shift when loading clips
    # batch_size = 64
    # epoch_ratio = 0.01 # to sample just a portion of the dataset
    # epochs = 10
    # lr = 0.0001
    # num_workers = 12

    # dim_model = 256
    # num_heads = 8
    # num_encoder_layers = 6
    # num_decoder_layers = 6
    # dropout_p = 0.1

    frames_per_clip = wandb.config.frames_per_clip
    frames_to_predict = wandb.config.frames_to_predict
    stride = wandb.config.stride # number of frames to shift when loading clips
    batch_size = wandb.config.batch_size
    epoch_ratio = wandb.config.epoch_ratio # to sample just a portion of the dataset
    epochs = wandb.config.epochs
    lr = wandb.config.lr
    num_workers = wandb.config.num_workers
    
    dim_model = wandb.config.dim_model
    num_heads = wandb.config.num_heads
    num_encoder_layers = wandb.config.num_encoder_layers
    num_decoder_layers = wandb.config.num_decoder_layers
    dropout_p = wandb.config.dropout_p

    use_mse = wandb.config.use_mse
    use_gdl = wandb.config.use_gdl
    lambda_gdl = wandb.config.lambda_gdl
    alpha = wandb.config.alpha
    frame_size = wandb.config.frame_size
    fps = wandb.config.frame_rate
    use_contrastive = wandb.config.use_contrastive
    lambda_contrastive = wandb.config.lambda_contrastive

    trainer = Trainer()
    model = Transformer(num_tokens=0, dim_model=dim_model, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout_p=dropout_p)
    
    print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume:
        model.load_state_dict(torch.load('./checkpoints/' + args.old_name + '.pt'))

    opt = optim.Adam(model.parameters(), lr=lr)
    # opt = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=15, num_training_steps=epochs*len(train_loader))
    scheduler = None
    # loss_fn = nn.MSELoss()
    loss_fn = trainer.criterion(use_mse=use_mse, use_gdl=use_gdl, lambda_gdl=lambda_gdl, alpha=alpha, use_contrastive=use_contrastive, lambda_contrastive=lambda_contrastive)  # , temperature)

    if 'ucf' in args.dataset:
        if args.dataset.endswith('wallpushups'):
            ucf_data_dir = 'data/UCF-101/UCF-101-wallpushups'
        elif args.dataset.endswith('workout'):
            ucf_data_dir = 'data/UCF-101/UCF-101-workout'
        elif args.dataset.endswith('instruments'):
            ucf_data_dir = 'data/UCF-101/UCF-101-instruments'
        elif args.dataset == 'ucf':
            ucf_data_dir = 'data/UCF-101/UCF-101'
        else:
            raise ValueError('Invalid dataset name')
            
        ucf_label_dir = 'data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'

        # ucf_data_dir = "/Users/jsikka/Documents/UCF-101"
        # ucf_label_dir = "/Users/jsikka/Documents/ucfTrainTestlist"
        

        tfs = transforms.Compose([
                # scale in [0, 1] of type float
                # transforms.Lambda(lambda x: x / 255.),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                # rescale to the most common size

                # transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
                transforms.Lambda(lambda x: nn.functional.interpolate(x, (frame_size, frame_size))),
                # horizontal flipping
                transforms.Lambda(lambda x: torch.flip(x, dims=[3]) if (torch.rand(1) > 0.5 and args.flip) else x),
                transforms.Lambda(lambda x: x.permute(0, 2, 3, 1)),
                # rgb to bgr
                transforms.Lambda(lambda x: x[..., [2, 1, 0]]),
                
        ])

        print('Loading UCF dataset from', ucf_data_dir)


        train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, train=True, transform=tfs, num_workers=num_workers, frame_rate=fps) # frames_between_clips/frame_rate
        print("Number of training samples: ", len(train_dataset))

        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, # shuffle=True
                                                collate_fn=trainer.custom_collate, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
        # create test loader (allowing batches and other extras)
        test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, train=False, transform=tfs, num_workers=num_workers, frame_rate=fps) # frames_between_clips/frame_rate
        print("Number of test samples: ", len(test_dataset))

        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * epoch_ratio))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, # shuffle=True
                                                collate_fn=trainer.custom_collate, num_workers=num_workers, pin_memory=True, sampler=test_sampler)
        
    elif args.dataset == 'ball':
        train_dataset = BouncingBall(num_frames=5, stride=stride, dir=args.folder, stage='train', shuffle=True)
        train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(len(train_dataset) * epoch_ratio))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        
        test_dataset = BouncingBall(num_frames=5, stride=stride, dir=args.folder, stage='test', shuffle=True)
        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * epoch_ratio))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        

    # print("TEST LOADER")
    # # print(test_loader)
    # for i in test_loader:
    #     print(i.size())
    #     print(i)
    #     break

    # opt = optim.Adam(model.parameters(), lr=lr)
    # # opt = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=15, num_training_steps=epochs*len(train_loader))
    # # loss_fn = nn.MSELoss() # TODO: change this to mse + contrastive + gradient difference
    # loss_fn = trainer.criterion(use_mse, use_gdl, lambda_gdl, alpha, use_contrastive)  # , temperature)

    wandb.run.name = args.config + '_' + str(trainer.index)

    best_train_loss = 1e10
    best_val_loss = 1e10
    epoch = 1
    
    # while True:
    for epoch in range(1, epochs+1):
        print("-"*25, f"Epoch {epoch}","-"*25)
        train_loss, validation_loss = trainer.fit(model=model, opt=opt,  scheduler=scheduler, loss_fn=loss_fn, train_dataloader=train_loader, val_dataloader=test_loader, frames_to_predict=frames_to_predict)
        
        if args.save_best: # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), './checkpoints/' + args.config + '_' + str(trainer.index) + '_train' + '.pt')
                print('model saved as ' + args.config + '_' + str(trainer.index)+ '_train.pt (best train loss)')
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                torch.save(model.state_dict(), './checkpoints/' + args.config + '_' + str(trainer.index) + '_test' + '.pt')
                print('model saved as ' + args.config + '_' + str(trainer.index) +  '_test.pt (best test loss)')
        else: # save last model
            torch.save(model.state_dict(), './checkpoints/' + args.config + '_' + str(trainer.index) + '.pt')
            print('model saved as ' + args.config + '_' + str(trainer.index) + '.pt')

if __name__ == '__main__':
    config, args = parse_config_args()

    # SET HYPERPARAMETERS HERE
    sweep_config = {
        'method': 'grid',
            }
    metric = {
            'name': 'val_loss',
            'goal': 'minimize'
        }
    sweep_config['metric'] = metric
    parameters_dict = {
        'frames_per_clip': {
            'values': config.FRAMES_PER_CLIP
        },
        'frames_to_predict': {
            'values': config.FRAMES_TO_PREDICT
        },
        'stride': {
            'values': config.STRIDE
        },
        'batch_size': {
            'values': config.BATCH_SIZE
        },
        'epoch_ratio': {
            'values': config.EPOCH_RATIO
        },
        'epochs': {
            'values': config.EPOCHS
        },
        'lr': {
            'values': config.LR
        },
        'num_workers': {
            'values': config.NUM_WORKERS
        },

        'dim_model': {
            'values': config.DIM_MODEL
        },
        'num_heads': {
            'values': config.NUM_HEADS
        },
        'num_encoder_layers': {
            'values': config.NUM_ENCODER_LAYERS
        },
        'num_decoder_layers': {
            'values': config.NUM_DECODER_LAYERS
        },
        'dropout_p': {
            'values': config.DROPOUT_P
        },
        'use_mse': {
            'values': config.USE_MSE
        },
        'use_gdl': {
            'values': config.USE_GDL
        },
        'lambda_gdl': {
            'values': config.LAMBDA_GDL
        },
        'alpha': {
            'values': config.ALPHA
        },
        'frame_size': {
            'values': [config.FRAME_SIZE]
        },
        'frame_rate': {
            'values': config.FPS
        },
        'use_contrastive': {
            'values': config.USE_CONTRASTIVE
        },
        'lambda_contrastive': {
            'values': config.LAMBDA_CONTRASTIVE
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='sd-video-gen', entity='sd-video-gen')

    wandb.agent(sweep_id, main)
    # wandb.agent(sweep_id, main, count=20) 