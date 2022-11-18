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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--index', type=str, required=True)
    # parser.add_argument('--pred_frames', type=int, default=1) # number of frames to predict
    # parser.add_argument('--show', type=bool, default=False)
    # parser.add_argument('--folder', type=str, required=True)
    # parser.add_argument('--name', type=str, default='default')
    # parser.add_argument('--fullscreen', type=bool, default=False)
    # parser.add_argument('--dataset', type=str, required=True)
    # args = parser.parse_args()

    config, args = parse_config_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.load_state_dict(torch.load('./checkpoints/' + str(args.config) + '.pt'))
    model.eval()
    model = model.to(device)


    if args.dataset == 'ball':
        test_dataset = BouncingBall(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    elif args.dataset == 'ucf':
        tfs = transforms.Compose([
                    # scale in [0, 1] of type float
                    # transforms.Lambda(lambda x: x / 255.),
                    # reshape into (T, C, H, W) for easier convolutions
                    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                    # rescale to the most common size

                    # transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
                    transforms.Lambda(lambda x: nn.functional.interpolate(x, (64, 64))),
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

        # ucf_data_dir = 'data/UCF-101/UCF-101'
        ucf_data_dir = 'data/UCF-101/UCF-101-wallpushups'
        ucf_label_dir = 'data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'
        test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=5, train=False, transform=tfs, num_workers=12) # frames_between_clips/frame_rate
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=12, pin_memory=True)
        
    with torch.no_grad():
        for index_list, batch in test_loader:
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
                    print('inputs shape: ', inputs.shape)

            for iteration in range(args.pred_frames):
                pred = predict(model, X)
                pred = torch.tensor(pred, dtype=torch.float32, device=device)
                preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), dim=1)
                print('preds shape: ', pred.shape)
                all_latents = torch.cat([inputs[:,:-1], preds], dim=1) # remove last input frame and add preds
                is_pred = [False] * (inputs.shape[1] - 1) + [True] * preds.shape[1]
                print('all_latents shape: ', all_latents.shape)
                X = all_latents[:, -5:] # the next input is the last 5 frames of the concatenated inputs and preds
                print('X after modifying: ', X.shape)

            if args.show:
                for i, latent in enumerate(all_latents.squeeze(0)):
                    latent = latent.reshape((1, 4, 8, 8))
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


    #     # counting number of files in ./checkpoints
    #     folder_index = len(os.listdir('./images'))   
    #     os.mkdir('./images/' + str(folder_index))
    #     img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')
    #     pred_img[0].save(img_path)
    #     # pred_img[0].show()