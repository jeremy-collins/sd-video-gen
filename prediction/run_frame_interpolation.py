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
created_dirs = []
for  r, d, f in os.walk(file_path):
        for counter in video_ids:
                  #f = f.sort()[:5]

                  for file in f:
                        if file.startswith(f'{counter}' + '_'):

                                dir_name = 'counter_' + f'{counter}'
                                created_dirs.append(dir_name)
                                if not os.path.exists( os.path.join(op_path, dir_name)):
                                        os.makedirs(os.path.join(op_path,dir_name))
                                for s in ['8.png', '9.png', '10.png', '11.png', '12.png']:
                                        if file.endswith(s):
                                            shutil.copy(os.path.join(r,file), os.path.join(op_path,dir_name ))

for dir_name in created_dirs:
        os.system('python3 -m frame_interpolation.eval.interpolator_cli --pattern "{}" --model_path frame_interpolation/pretrained_models/saved_model --times_to_interpolate 2 --output_video'.format(os.path.join(op_path, dir_name)))





