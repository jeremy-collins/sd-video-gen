# sd-video-gen

## Setup
- Clone the repository with the command `git clone https://github.com/jeremy-collins/sd-video-gen.git`
- Install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so already.
- Install the dependencies: `conda env create --name ls_project --file environment.yml`

## Training a model
- `python trainer.py --dataset <dataset_name>`, where dataset_name is `ball` for the bouncing ball dataset or `ucf` for the UCF-101 dataset.

## Running a trained model
- `python predict.py --index <index>`, where index is the name of the model checkpoint. The naming convention is `model_<index>` in the checkpoints folder.
