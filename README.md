# Video Generation using Stable Diffusion

![generated video](https://github.com/jeremy-collins/sd-video-gen/blob/b5842d92f4189e4765394614808e9c4487bd1003/gifs/ball_8.gif)
![generated video](https://github.com/jeremy-collins/sd-video-gen/blob/b5842d92f4189e4765394614808e9c4487bd1003/gifs/kitti_6.gif)

## Setup
- Clone the repository with the command `git clone https://github.com/jeremy-collins/sd-video-gen.git`
- Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so already.
- Install the dependencies: `conda env create --name ls_project --file environment.yml`
- Create relevant directories: `mkdir data` `mkdir checkpoints`
- Download model checkpoints [here](https://gtvault-my.sharepoint.com/:f:/g/personal/sganesh68_gatech_edu/Esz3KxV2S5RHnlD1hizh8X4B7RuEE3CI49nyavdfzPpgTA?e=rYPomk) and place them in the checkpoints folder

## Datasets
- To download one of the bouncing ball datasets, choose a zip file from [here](https://gtvault-my.sharepoint.com/:f:/g/personal/sganesh68_gatech_edu/ElJN29JY6d9PpxFIqu1IY5IB9nUG4-baHhX8_m6dBW58_w?e=CoByo5)
- Download the original UCF-101 dataset [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar), or download the formatted UCF-101 dataset from [here](https://google.com)
  - You will need to `pip install av` to load the UCF dataset.
- Unzip chosen datasets and place them in the data folder.

## Training a model
- `python -m trainers.trainer --dataset <ball, ucf, ucf-instruments, ucf-wallpushups, ucf-workout> --config <config>`
- Optional args:
  - `--save_best True` To save only the model with the lowest validation loss.
  - `--resume True --old_name <old model name>` to resume training from a previous checkpoint.
  - `--debug True` to turn off wandb logging.
  - `--flip True` to turn on horizontal flipping augmentation.
## Running a trained model
- `python -m prediction.predict --pred_frames <number of predicted frames to show> --dataset <ball, ucf, ucf-instruments, ucf-wallpushups, ucf-workout> --config <config> --index <number at the end of the model name>`
- Optional args:
  - `--folder <data folder>` to specify a data folder. Must contain train and test subfolders.
  - `--show True` to view ground truth and predicted video frames.
  - `--fullscreen True` to view the frames as fullscreen images .
  - `--mode <Train/Test>` to evaluate on the corresponding partition of the dataset.
  - `--denoise True` to denoise the generated frames using the pre-trained Stable Diffusion U-Net.
  - `--denoise_start_step <int between 0 and 50>` to control how much the generated frames are denoised. 0 = image from scratch, 50 = no denoising
  - `--save_output True` to save ground truth and generated frames to your files.
  
- NOTE: You will need to be logged into HuggingFace to run the Stable Diffusion models. You can log in via the terminal by echoing your HF token as an environment variable: `export HF_TOKEN=<token>`