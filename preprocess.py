import os
import cv2
import torch
import numpy as np
import argparse

from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

def preprocess(folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    auth_token = os.environ['HF_TOKEN']
    print('loading VAE...')
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
    vae = vae.to(device)
    
    for dir, _, files in tqdm(os.walk(folder)):
        for file in files:
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(dir, file))
                latents = encode_img(vae, img)
                latents = latents.detach().cpu().numpy()
                np.save(os.path.join(dir, file[:-4]), latents)

                

def encode_img(vae, imgs):
        # turn an image into image latents
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(imgs, list):
            imgs = [imgs]

        img_arr = np.stack([np.array(img) for img in imgs], axis=0)
        img_arr = img_arr / 255.0
        img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = vae.encode(img_arr.to(device))
        latent_samples = latent_dists.sample()
        latent_samples *= 0.18215

        return latent_samples 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    preprocess(args.folder)