import torch
import torch.nn as nn
import numpy as np
from transformer import Transformer
from bouncing_ball_loader import BouncingBall
from sd_utils import SDUtils
import PIL
import cv2
import os
import argparse
from torchvision.datasets import UCF101
import torchvision.transforms as transforms
from config import parse_config_args
from torch.utils.data import DataLoader, RandomSampler
import fvd

def predict(model, input_sequence):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():

        y_input = input_sequence # should have sos if trained with it
        # Get target mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask) # (batch_size, seq_len, dim_model)
        
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
    
if __name__ == "__main__":  
    config, args = parse_config_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: make config make sense
    model = Transformer(num_tokens=0, dim_model=config.DIM_MODEL[0], num_heads=config.NUM_HEADS[0], num_encoder_layers=config.NUM_ENCODER_LAYERS[0], num_decoder_layers=config.NUM_DECODER_LAYERS[0], dropout_p=config.DROPOUT_P[0]) 
    model.load_state_dict(torch.load('./checkpoints/' + str(args.config) + '_' + str(args.index)+ '_' + str(args.mode) + '.pt'))
    model.eval()
    model = model.to(device)


    if args.dataset == 'ball':
        test_dataset = BouncingBall(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    elif 'ucf' in args.dataset:
        if args.dataset.endswith('wallpushups'):
            ucf_data_dir = '../dataset/UCF-101/UCF-101-wallpushups'
        elif args.dataset.endswith('workout'):
            ucf_data_dir = '../dataset/UCF-101/UCF-101-workout'
        elif args.dataset.endswith('instruments'):
            ucf_data_dir = '../dataset/UCF-101/UCF-101-instruments'
        elif args.dataset == 'ucf':
            ucf_data_dir = '../dataset/UCF-101/UCF-101'
        else:
            raise ValueError('Invalid dataset name')
            
        ucf_label_dir = '../dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'

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
            test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=5, train=True, transform=tfs, num_workers=config.NUM_WORKERS[0], frame_rate=3) # frames_between_clips/frame_rate
            # ***TRAIN***

        else:
            # ***TEST***
            test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=9, train=False, transform=tfs, num_workers=config.NUM_WORKERS[0]) # frames_between_clips/frame_rate
            # ***TEST***

        print('epoch', config.EPOCH_RATIO[0])
        test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=int(len(test_dataset) * config.EPOCH_RATIO[0]))#config.EPOCH_RATIO[0]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=custom_collate, num_workers=config.NUM_WORKERS[0], pin_memory=True)

    dectector_for_fvd = fvd.load_detector()
    stats_groundtruth = fvd.get_FeatureStats()
    stats_predicted = fvd.get_FeatureStats()

    dummy = True
        
    with torch.no_grad():
        for index_list, batch in test_loader:
            print('index_list', index_list)
            inputs = torch.tensor([], device=device)
            preds = torch.tensor([], device=device)
            is_pred = []

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

            trans224 = transforms.Compose([transforms.Lambda(lambda x: x.permute(0, 1, 4, 2, 3)),
                                            transforms.Lambda(lambda x: x.squeeze(0)), 
                                            transforms.Resize(224), 
                                            transforms.Lambda(lambda x: x.unsqueeze(0)),
                                            transforms.Lambda(lambda x: x.permute(0, 1, 3, 4, 2)),
                                            ])
            groundtruth_frames = trans224(batch)
            #print('groundtruth_frames', groundtruth_frames.shape)
            fvd.update_stats_for_sequence(dectector_for_fvd, stats_groundtruth, groundtruth_frames)


            for iteration in range(args.pred_frames):
                pred = predict(model, X)
                if args.denoise:
                    #print('denoising predicted frame...')
                    #print('pred.shape:', pred.shape)
                    uncond_text_embeddings = sd_utils.encode_text([''])
                    denoise_pred = pred.reshape((1, 4, config.FRAME_SIZE // 8, config.FRAME_SIZE // 8))
                    # interpolate to 64x64
                    denoise_pred = nn.functional.interpolate(denoise_pred, (64, 64), mode='bilinear') # resize to latents corresponding to 512x512 image
                    # denoise_pred = sd_utils.denoise_img_latents(text_embeddings=uncond_text_embeddings, height=config.FRAME_SIZE, width=config.FRAME_SIZE, latents=denoise_pred, num_inference_steps=100, guidance_scale=0)
                    denoise_pred = sd_utils.gen_i2i_latents(uncond_text_embeddings, height=config.FRAME_SIZE, width=config.FRAME_SIZE,
                                    num_inference_steps=50, guidance_scale=0, latents=denoise_pred,
                                    return_all_latents=False, start_step=48)
                    denoised_img = sd_utils.decode_img_latents(denoise_pred)
                    denoised_img = torch.tensor(denoised_img, device=device) # .permute(0, 3, 1, 2)
                    #print('denoised_img.shape:', denoised_img.shape)
                    denoised_img = nn.functional.interpolate(denoised_img.permute(0, 3, 1, 2), (config.FRAME_SIZE, config.FRAME_SIZE))
                    denoised_img = denoised_img.permute(0, 2, 3, 1).unsqueeze(0)
                    pred = sd_utils.encode_batch(denoised_img, use_sos=False)
                    #print('pred.shape:', pred.shape)
                    pred = pred.flatten()

                pred = torch.tensor(pred, dtype=torch.float32, device=device)
                preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), dim=1)
                #print('preds shape: ', pred.shape)

                
 
                all_latents = torch.cat([inputs[:,:-1], preds], dim=1) # remove last input frame and add preds
                is_pred = [False] * (inputs.shape[1] - 1) + [True] * preds.shape[1]
                #print('all_latents shape: ', all_latents.shape)
                X = all_latents[:, -5:] # the next input is the last 5 frames of the concatenated inputs and preds
                #print('X after modifying: ', X.shape)

                
            predicted_frames = []
            if args.save_output:
                frame_indices = index_list[0]
                for i, latent in enumerate(all_latents.squeeze(0)):
                    # latent = latent.reshape((1, 4, 8, 8))
                    latent = latent.reshape((1, 4, config.FRAME_SIZE // 8, config.FRAME_SIZE // 8))
                    img = sd_utils.decode_img_latents(latent)
                    img = np.array(img[0])
                    
                    if is_pred[i]:
                        # add a red border to the predicted frames
                        #img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                        # save to args.folder/results/<4 digit folder ID + 3 digit file/frame ID>.png

                        pr_val = cv2.imwrite(os.path.join('outputs', str(args.config) + '_' + str(args.index)+ '_' + str(args.mode), str(index_list.item()) + '_' + str(i) + '.png'), img)
                        #if(dummy or not pr_val):
                            #print(os.path.join('outputs', str(args.config) + '_' + str(args.index)+ '_' + str(args.mode), str(index_list.item()) + '_' + str(i) + '.png'))
                            #dummy=False

                        predicted_frames.append(img)
                    # img_path = os.path.join('./images', str(folder_index), str(index_list[idx - 1].item()) + '_gt.png')
                    # input_img[0].save(img_path)
                    # cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow('frame', img)
                    # cv2.waitKey(0)
                predicted_frames = torch.from_numpy(np.array(predicted_frames).reshape((1,args.pred_frames,config.FRAME_SIZE, config.FRAME_SIZE, 3)))
                #print('predicted_frames', predicted_frames.shape)
                predicted_frames = trans224(predicted_frames)
                fvd.update_stats_for_sequence(dectector_for_fvd, stats_predicted, predicted_frames)

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
                    
        fvd_score = fvd.compute_fvd(stats_predicted, stats_groundtruth)
        print('FVD : ', fvd_score)

    #     # counting number of files in ./checkpoints
    #     folder_index = len(os.listdir('./images'))   
    #     os.mkdir('./images/' + str(folder_index))
    #     img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')
    #     pred_img[0].save(img_path)
    #     # pred_img[0].show()