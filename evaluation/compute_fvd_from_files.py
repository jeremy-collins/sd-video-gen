import torch
import torch.nn as nn
import numpy as np
from models.transformer import Transformer
from loaders.bouncing_ball_loader import BouncingBall
from utils.sd_utils import SDUtils
from PIL import Image
import cv2
import os
import argparse
from torchvision.datasets import UCF101
import torchvision.transforms as transforms
from utils.config import parse_config_args
from fvd_2 import get_fvd_logits, frechet_distance, load_i3d_pretrained, all_gather
from torch.utils.data import DataLoader, RandomSampler
    
if __name__ == "__main__":  
    #config, args = parse_config_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: make config make sense

    i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        real_embeddings = []
        fake_embeddings = []
        real_input = None
        fake_input = None
        i3d = load_i3d_pretrained(device)
        #for (ind, (index_list, batch)) in enumerate(test_loader):
        for batch_ind in range(0, 128):
            image_batch = torch.zeros((16, 15, 128, 128, 3))
            curr_batch_start = batch_ind * 16
            for seq_num in range(curr_batch_start, curr_batch_start+16):
                for frame_num in range(0, 15):
                    if os.path.exists('real_frames/' + str(seq_num) + '_' + str(frame_num) + '.png'):
                        image = Image.open('real_frames/' + str(seq_num) + '_' + str(frame_num) + '.png')
                        transform = transforms.Compose([
                            transforms.PILToTensor()
                        ])
                        img_tensor = transform(image).permute(1,2,0)
                        image_batch[(seq_num - curr_batch_start), frame_num, :, :, :] = img_tensor

            print("image_batch", image_batch.shape)

            real_input = image_batch #.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
            real = (real_input * 255).numpy().astype('uint8')
            real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))
            print('real_embeddings len', len(real_embeddings))

            pred_image_batch = torch.zeros((16, 15, 128, 128, 3))
            for seq_num in range(curr_batch_start, curr_batch_start+16):
                for frame_num in range(0, 15):
                    if os.path.exists('predicted_images/counter_' + str(seq_num) + '/interpolated_frames/frame_' + str(frame_num).zfill(3) + '.png'):
                        image = Image.open('predicted_images/counter_' + str(seq_num) + '/interpolated_frames/frame_' + str(frame_num).zfill(3) + '.png')
                        transform = transforms.Compose([
                            transforms.PILToTensor()
                        ])
                        img_tensor = transform(image).permute(1,2,0)
                        pred_image_batch[(seq_num - curr_batch_start), frame_num, :, :, :] = img_tensor            
            
            fake_input = pred_image_batch
            #fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
            fake = (fake_input * 255).numpy().astype('uint8')
            fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
            fake_input = None
            print('fake_embeddings len', len(fake_embeddings))

        fake_embeddings = torch.cat(fake_embeddings)
        real_embeddings = torch.cat(real_embeddings)

        print('fake_embeddings shape', fake_embeddings.shape)
        print('real_embeddings shape', real_embeddings.shape)

        fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
        print("FVD: ", fvd)
