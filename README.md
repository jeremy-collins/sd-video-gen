# sd-video-gen

## Setup
- Clone the repository with the command `git clone https://github.com/jeremy-collins/sd-video-gen.git`
- Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so already.
- Install the dependencies: `conda env create --name ls_project --file environment.yml`

## Training a model
- `python trainer.py --dataset <dataset_name> --save_best <T/F>', where dataset_name is `ball` for the bouncing ball dataset or `ucf` for the UCF-101 dataset, and when save_best is True, only the model with the lowest validation loss will be saved.

## Running a trained model
- `python predict.py --index <index> --pred_frames <pred_frames>, --show <T/F>`, where index is the name of the model checkpoint (naming convention is `model_<index>` in the checkpoints folder), pred_frames is the number of frames to autoregressively predict, and show determines if the images will be viewed.
