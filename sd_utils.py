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



# Stable Diffusion utils
class SDUtils():
  def __init__(self):
    
    # pipe, vae, tokenizer, text_encoder, unet, scheduler = load_models()
    vae = self.load_models()
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.pipe = pipe
    self.vae = vae
    # self.tokenizer = tokenizer
    # self.text_encoder = text_encoder
    # self.unet = unet
    # self.scheduler = scheduler
    
    
  def load_models(self):
    auth_token = os.environ['HF_TOKEN']
    # initializing entire pipeline for out of the box SD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pipe = StableDiffusionPipeline.from_pretrained(
    #     'CompVis/stable-diffusion-v1-4', revision='fp16',
    #     torch_dtype =torch.float16, use_auth_token=True)
    # pipe = pipe.to(device)


    print('loading VAE...')
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
    vae = vae.to(device)


    # print('loading tokenizer...')
    # # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    # tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    # text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    # text_encoder = text_encoder.to(device)

    # print('loading UNet...')
    # # 3. The UNet model for generating the latents.
    # unet = UNet2DConditionModel.from_pretrained(
    #     'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
    # unet = unet.to(device)

    # # 4. Create a scheduler for inference
    # scheduler = LMSDiscreteScheduler(
    #     beta_start=0.00085, beta_end=0.012,
    #     beta_schedule='scaled_linear', num_train_timesteps=1000)
    
    return vae
      
  def encode_text(self, prompt):
    # Tokenize a prompt or a list of prompts and get embeddings
    text_input = self.tokenizer(
        prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    with torch.no_grad():
      text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

    # Do the same for unconditional embeddings
    uncond_input = self.tokenizer(
        [''] * len(prompt), padding='max_length',
        max_length=self.tokenizer.model_max_length, return_tensors='pt')
    with torch.no_grad():
      uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

  def denoise_img_latents(self, text_embeddings, height=512, width=512,
                      num_inference_steps=50, guidance_scale=7.5, latents=None, start_step=None):
    # apply text conditioned denoising steps, starting with noise
    if latents is None:
      latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, \
                            height // 8, width // 8))
    latents = latents.to(self.device)

    self.scheduler.set_timesteps(num_inference_steps)
    latents = latents * self.scheduler.sigmas[0]

    with autocast('cuda'):
      for i, t in tqdm(enumerate(self.scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = self.scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        with torch.no_grad():
          noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, i, latents)['prev_sample']
    
    return latents
   
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

  def decode_img_latents(self, latents):
    # get images from image latents

    latents = 1 / 0.18215 * latents

    with torch.no_grad():
      imgs = self.vae.decode(latents)

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images

  def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50,
                  guidance_scale=7.5, latents=None):
    # go from a prompt or a list of prompts to generate image(s)

    if isinstance(prompts, str):
      prompts = [prompts]

    # Prompts -> text embeddings
    text_embeds = self.encode_text(prompts)

    # Text embeds -> img latents
    latents = self.denoise_img_latents(
        text_embeds, height=height, width=width, latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    # Img latents -> imgs
    imgs = self.decode_img_latents(latents)

    return imgs

  def imgs_to_video(self, imgs, video_name='video.mp4', fps=15):
    # turn list of images into an mp4
    video_dims = (imgs[0].width, imgs[0].height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')    
    video = cv2.VideoWriter(video_name, fourcc, fps, video_dims)
    for img in imgs:
      tmp_img = img.copy()
      video.write(cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR))
    video.release()

  def display_video(self, file_path, width=512):
    # play an mp4 file in colab
    compressed_vid_path = 'comp_' + file_path
    if os.path.exists(compressed_vid_path):
      os.remove(compressed_vid_path)
    os.system(f'ffmpeg -i {file_path} -vcodec libx264 {compressed_vid_path}')

    mp4 = open(compressed_vid_path, 'rb').read()
    data_url = 'data:simul2/mp4;base64,' + b64encode(mp4).decode()
    return HTML("""
      <video width={} controls>
            <source src="{}" type="video/mp4">
      </video>
      """.format(width, data_url))
    
  def perturb_latents(self, latents, scale=0.1):
    # jitter the image latents. Denoise and decode this to generate variations
    noise = torch.randn_like(latents)
    new_latents = (1 - scale) * latents + scale * noise
    return (new_latents - new_latents.mean()) / new_latents.std()
    
  def gen_i2i_latents(self, text_embeddings, height=512, width=512,
                    num_inference_steps=50, guidance_scale=7.5, latents=None,
                    return_all_latents=False, start_step=10):
    # create a noised version of the input image latents so we can denoise it to create an image conditioned on the input image
    # for usage in img_to_img()
    if latents is None:
      latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, \
                            height // 8, width // 8))
    latents = latents.to(self.device)

    # New scheduler for img-to-img
    scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012,
    beta_schedule='scaled_linear', num_train_timesteps=1000)
    
    scheduler.set_timesteps(num_inference_steps)
    if start_step > 0:
      start_timestep = scheduler.timesteps[start_step]
      start_timesteps = start_timestep.repeat(latents.shape[0]).long()

      noise = torch.randn_like(latents)
      latents = scheduler.add_noise(latents, noise, start_timesteps)

    latent_hist = [latents]
    with autocast('cuda'):
      for i, t in tqdm(enumerate(scheduler.timesteps[start_step:])):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        # predict the noise residual
        with torch.no_grad():
          noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']
        latent_hist.append(latents)
    
    if not return_all_latents:
      return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents

  def img_to_img(self, prompts, height=512, width=512, num_inference_steps=50,
                  guidance_scale=7.5, img=None, return_all_latents=False,
                  batch_size=2, start_step=10):
    # image generation conditioned on an input image
    if isinstance(prompts, str):
      prompts = [prompts]

    # input image -> img latents
    input_img_latents = self.encode_img(img)

    # Prompts -> text embeds
    text_embeds = self.encode_text(prompts)

    # Text embeds -> img latents
    i2i_latents = self.gen_i2i_latents(
        text_embeds, height=height, width=width, latents=input_img_latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        return_all_latents=return_all_latents, start_step=start_step)
    
    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(i2i_latents), batch_size)):
      imgs = self.decode_img_latents(i2i_latents[i:i+batch_size])
      all_imgs.extend(imgs)

    return all_imgs

  