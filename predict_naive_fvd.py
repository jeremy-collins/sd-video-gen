import torch
import torch.nn as nn
import numpy as np
from identity import Identity
from bouncing_ball_loader import BouncingBall
from sd_utils import SDUtils
import PIL
import cv2
import os
import argparse
from torchvision.datasets import UCF101
import torchvision.transforms as transforms
from config import parse_config_args
from fvd_2 import get_fvd_logits, frechet_distance, load_i3d_pretrained, all_gather
from torch.utils.data import DataLoader, RandomSampler

def predict(model, input_sequence, cls_list):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():

        y_input = input_sequence # should have sos if trained with it
        # Get target mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, cls_list, y_input, tgt_mask) # (batch_size, seq_len, dim_model)
        
        # Permute pred to have batch size first again
        pred = pred.permute(1, 0, 2)
        # new shape: (batch_size, seq_len, dim_model)
        
        # X shape is (batch_size, src sequence length, input.shape)
        # y_input shape is (batch_size, tgt sequence length, input.shape)
    
        # next item is the last item in the predicted sequence
        next_item = pred[:,-1,:].unsqueeze(1)
        # next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with prediction
        y_input = torch.cat((y_input, next_item), dim=1)

    return pred[0, -1] # return last item in sequence
def find_classes(directory): # -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    classes = splitClassNames(classes)
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return classes, idx_to_class

def splitClassNames(classes):
    result = []
    for s in classes:
        new_string=""
        for i in s:
            if(i.isupper()):
                new_string+="*"+i
            else:
                new_string+=i
        x=new_string.split("*")
        x.remove('')
        print(x)
        x = " ".join(x)
        result.append(x)

    return result
    
if __name__ == "__main__":  
    config, args = parse_config_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: make config make sense
    model = Identity()
    #model.load_state_dict(torch.load('./checkpoints/' + str(args.config) + '_' + str(args.index)+ '_' + str(args.mode) + '.pt'))
    model.eval()
    model = model.to(device)
    i3d = load_i3d_pretrained(device)
    idx_to_class = None

    if args.dataset == 'ball':
        test_dataset = BouncingBall(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    elif 'ucf' in args.dataset:
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

        classes, idx_to_class = find_classes(ucf_data_dir)
        ucf_label_dir = 'data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'
        classes, idx_to_class = find_classes(ucf_data_dir)

        tfs = transforms.Compose([
                # scale in [0, 1] of type float
                # transforms.Lambda(lambda x: x / 255.),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                # rescale to the most common size

                # transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
                transforms.Lambda(lambda x: nn.functional.interpolate(x, (config.FRAME_SIZE, config.FRAME_SIZE))),
                transforms.Lambda(lambda x: x.permute(0, 2, 3, 1)),
                # rgb to bgr
                transforms.Lambda(lambda x: x[..., [2, 1, 0]]),
        ])


        def custom_collate(batch):
            filtered_batch = []
            for video, _, label in batch:
                # filtered_batch.append((video, label))
                filtered_batch.append((label, video))
            return torch.utils.data.dataloader.default_collate(filtered_batch)


        print('Loading UCF dataset from', ucf_data_dir)
        
        if args.mode == 'train':
            # ***TRAIN***
            test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=5, train=True, transform=tfs, num_workers=12, frame_rate=3) # frames_between_clips/frame_rate
            # ***TRAIN***

        else:
            # ***TEST***
            test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=16, train=False, transform=tfs, num_workers=12) # frames_between_clips/frame_rate
            # ***TEST***

        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=2048)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=custom_collate, num_workers=12, pin_memory=True)
        
    with torch.no_grad():
        real_embeddings = []
        fake_embeddings = []
        real_input = None
        fake_input = None
        i3d = load_i3d_pretrained(device)
        for (ind, (index_list, batch)) in enumerate(test_loader):
            if ind > 512:
                break
            #print('index_list', index_list)
            inputs = torch.tensor([], device=device)
            preds = torch.tensor([], device=device)
            is_pred = []

            cls_list = None
            if idx_to_class is not None:
                # print(idx_to_class.keys())
                cls_list =[idx_to_class[idx_] for idx_ in index_list.tolist()]

            #print("batch", batch.shape)
            #real = batch.reshape()

            real = batch #.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
            if(real_input == None):
                real_input = real
            else:
                real_input = torch.cat((real_input, real), 0)
            #print("real_input.shape", real_input.shape)
            if(real_input.shape[0] >= 16):
                real = (real_input * 255).numpy().astype('uint8')
                real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))
                real_input = None
                print('real_embeddings len', len(real_embeddings))

            new_batch = sd_utils.encode_batch(batch, use_sos=True)
            new_batch = torch.tensor(new_batch).to(device)

            X = new_batch
            y = new_batch

            X = torch.tensor(X, dtype=torch.float32, device=device)

            # shift the tgt by one so with the <SOS> we predict the token at pos 1
            y = torch.tensor(y[:,:-1], dtype=torch.float32, device=device)


            for idx, input in enumerate(X.squeeze(0)): # for each input frame
                if idx == 0:
                        continue # SOS token
                else:
                    inputs = torch.cat((inputs, input.unsqueeze(0).unsqueeze(0)), dim=1)
                    #print('inputs shape: ', inputs.shape)
            #print('denoising sequence ', ind)

            for iteration in range(args.pred_frames):
                pred = predict(model, X, cls_list)
                if args.denoise:
                    #print('denoising predicted frame...')
                    #print('pred.shape:', pred.shape)
                    uncond_text_embeddings = sd_utils.encode_text([idx_to_class[index_list[0].item()]])
                    noisy_pred = pred.reshape((1, 4, config.FRAME_SIZE // 8, config.FRAME_SIZE // 8)) # unflattening pred latent
                    #print('denoise_pred.shape:', noisy_pred.shape)

                    # decoding pred
                    noisy_img = sd_utils.decode_img_latents(noisy_pred) # shape (1, 3, FRAME_SIZE, FRAME_SIZE) # decoding into image
                    noisy_img = torch.tensor(noisy_img, device=device)
                    #print('noisy_img.shape:', noisy_img.shape)

                    # upscaling image
                    noisy_img = nn.functional.interpolate(noisy_img.permute(0, 3, 1, 2), (512, 512)) # upscaling predicted image to FRAME_SIZE
                    noisy_img = noisy_img.permute(0, 2, 3, 1).unsqueeze(0)
                    #print('upscaled noisy_img.shape:', noisy_img.shape)

                    # encoding upscaled image
                    resized_latents = sd_utils.encode_batch(noisy_img, use_sos=False) # encoding interpolated image
                    resized_latents = resized_latents.reshape((1, 4, 512//8, 512//8)) # reshaping to (1, 4, 512, 512)
                    #print('resized_latents.shape:', resized_latents.shape)

                    # denoising
                    denoised_latents = sd_utils.gen_i2i_latents(uncond_text_embeddings, height=512, width=512, # denoising interpolated image
                                    num_inference_steps=50, guidance_scale=7.5, latents=resized_latents,
                                    return_all_latents=False, start_step=args.denoise_start_step)

                    # decoding denoised image
                    denoised_img = sd_utils.decode_img_latents(denoised_latents) # shape (1, 3, 512, 512) # decoding into large image
                    denoised_img = torch.tensor(denoised_img, device=device)
                    #print('denoised_img.shape:', denoised_img.shape)

                    # downscaling denoised image
                    denoised_img = nn.functional.interpolate(denoised_img.permute(0, 3, 1, 2), (config.FRAME_SIZE, config.FRAME_SIZE)) # shrinking denoised image back to FRAME_SIZE
                    denoised_img = denoised_img.permute(0, 2, 3, 1).unsqueeze(0)
                    #print('shrunken denoised_img.shape:', denoised_img.shape)

                    # encoding shrunken denoised image
                    pred = sd_utils.encode_batch(denoised_img, use_sos=False) # encoding shrunken denoised image
                    #print('denoised_latents.shape:', denoised_latents.shape)
                    pred = pred.flatten()

                pred = torch.tensor(pred, dtype=torch.float32, device=device)
                preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), dim=1)
                #print('preds shape: ', pred.shape)

                
 
                all_latents = torch.cat([inputs[:,:-1], preds], dim=1) # remove last input frame and add preds
                is_pred = [False] * (inputs.shape[1] - 1) + [True] * preds.shape[1]
                #print('all_latents shape: ', all_latents.shape)
                X = all_latents[:, -5:] # the next input is the last 5 frames of the concatenated inputs and preds
                #print('X after modifying: ', X.shape)

            
            
            fake_curr_inp = None
            if args.save_output:
                frame_indices = index_list[0]
                
                for i, latent in enumerate(all_latents.squeeze(0)):
                    # latent = latent.reshape((1, 4, 8, 8))
                    latent = latent.reshape((1, 4, config.FRAME_SIZE // 8, config.FRAME_SIZE // 8))
                    img = sd_utils.decode_img_latents(latent)
                    img = np.array(img[0])
                    
                    if is_pred[i]:
                        #print('img', img.shape)
                        curr_frame = torch.from_numpy(img).unsqueeze(0)
                        if(fake_curr_inp == None):
                            fake_curr_inp = curr_frame
                        else:
                            fake_curr_inp = torch.cat((fake_curr_inp, curr_frame), 0)
                        #print("fake_curr_inp.shape", fake_curr_inp.shape)

                        pr_val = cv2.imwrite(os.path.join('outputs_pred', str(args.config) + '_' + str(args.index)+ '_' + str(args.mode) + '_naive', str(ind) + '_' + str(i) + '.png'), img)

                    else:
                        pr_val = cv2.imwrite(os.path.join('outputs_real', str(args.config) + '_' + str(args.index)+ '_' + str(args.mode) + '_naive', str(ind) + '_' + str(i) + '.png'), img)
                        
                fake_curr_inp = fake_curr_inp.unsqueeze(0)
                if fake_input == None:
                    fake_input = fake_curr_inp
                else:
                    fake_input= torch.cat((fake_input, fake_curr_inp), 0)
                #print("fake_input.shape", fake_input.shape)
                if fake_input.shape[0] >= 16:
                    #fake = torch.from_numpy(img)
                    #fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
                    fake = (fake_input * 255).numpy().astype('uint8')
                    fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
                    fake_input = None
                    print('fake_embeddings len', len(fake_embeddings))

            if args.show:
                for i, latent in enumerate(all_latents.squeeze(0)):
                    # latent = latent.reshape((1, 4, 8, 8))
                    latent = latent.reshape((1, 4, config.FRAME_SIZE // 8, config.FRAME_SIZE // 8))
                    img = sd_utils.decode_img_latents(latent)
                    img = np.array(img[0])
                    
                    if is_pred[i]:
                        # add a red border to the predicted frames
                        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                    # img_path = os.path.join('./images', str(folder_index), str(index_list[idx - 1].item()) + '_gt.png')
                    # input_img[0].save(img_path)
                    if args.fullscreen:
                        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('frame', img)
                    cv2.waitKey(0)
            #print('fake_embeddings len', len(fake_embeddings))
            #print('real_embeddings len', len(real_embeddings))
            if ind % 16 == 0:
                torch.save(real_embeddings, 'pickles/real'+args.config+'.pt')
                torch.save(fake_embeddings, 'pickles/fake'+args.config+'.pt')
                #real_embeddings = torch.load('pickles/real'+args.config+'.pt')
                #fake_embeddings = torch.load('pickles/fake'+args.config+'.pt')

        #fake_embeddings = all_gather(fake_embeddings)
        #real_embeddings = all_gather(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)
        real_embeddings = torch.cat(real_embeddings)

        print('fake_embeddings shape', fake_embeddings.shape)
        print('real_embeddings shape', real_embeddings.shape)

        fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
        print("FVD: ", fvd)

    #     # counting number of files in ./checkpoints
    #     folder_index = len(os.listdir('./images'))   
    #     os.mkdir('./images/' + str(folder_index))
    #     img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')
    #     pred_img[0].save(img_path)
    #     # pred_img[0].show()
