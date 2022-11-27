import os
import shutil
"""
predicted_images is the name of the folder containing predicted frames.

Steps to setup FILM:
Github FILM code is downloaded and placed in the same directory
https://github.com/google-research/frame-interpolation
Download pretrained models at https://drive.google.com/drive/folders/153dvxVSAcsNv1cyHVJySYZ-Twchm4Jdi

Run pip install -r requirements.txt
Then to resolve errors, run the following:
pip3 install dill==0.3.1.1
pip3 install requests==2.24.0
pip3 install protobuf==3.19.5

"""
file_path="predicted_images"
op_path="predicted_images"

video_ids = set()
for r,d,f in os.walk(file_path):
        for file in f:
                temp = file.split('_')
                video_ids.add(temp[0])

video_ids = list(video_ids)

for  r, d, f in os.walk(file_path):
        for counter in video_ids:

                for file in f:
                        if file.startswith(f'{counter}' + '_'):

                                dir_name = 'counter_' + f'{counter}'
                                if not os.path.exists( os.path.join(op_path, dir_name)):
                                        os.makedirs(os.path.join(op_path,dir_name))
                                shutil.copy(os.path.join(r,file), os.path.join(op_path,dir_name ))
                                os.system('python3 -m frame_interpolation.eval.interpolator_cli --pattern "{}" --model_path frame_interpolation/pretrained_models/saved_model --times_to_interpolate 2 --output_video'.format(dir_name))



