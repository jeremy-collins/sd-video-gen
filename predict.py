import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from transformer import Transformer
from bouncing_ball_loader import BouncingBall
from sd_utils import SDUtils
import PIL
import cv2
import os
import argparse

def predict(model, input_sequence, max_length=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    # y_input = torch.ones((1, model.dim_model), dtype=torch.float32, device=device) * 2 # SOS token
    SOS_token = torch.ones((1, 1, model.dim_model), dtype=torch.float32, device=device) * 2
    EOS_token = torch.ones((1, 1, model.dim_model), dtype=torch.float32, device=device) * 3
    # y_input = torch.tensor([SOS_token], dtype=torch.float32, device=device)
    y_input = SOS_token
    
    # num_tokens = len(input_sequence[0])
    with torch.no_grad():
        # for _ in range(6):
        # for _ in range(max_length):
        y_input = torch.cat((SOS_token, input_sequence), dim=1)
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

        # Stop if model predicts end of sentence
        # if next_item.view(-1).item() == EOS_token:
        #     break

    # return y_input.view(-1).tolist()
    # return pred[0,0].view(-1).tolist()
    return pred[0, -1] # return last item in sequence
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--pred_frames', type=int, default=1) # number of frames to predict
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--name', type=str, default='default')
    
    args = parser.parse_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.load_state_dict(torch.load('./checkpoints/model_' + str(args.index) + '.pt'))
    model.eval()
    model = model.to(device)
    
    # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='test', shuffle=True)
    # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_blackwhite1/content/2D-bouncing/test3_bouncing_ball', stage='test', shuffle=True)
    # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/tccvg/bouncing_ball_3000_blackwhite_simple1/content/2D-bouncing/test2_simple_bouncing_ball', stage='test', shuffle=True)
    test_dataset = BouncingBall(num_frames=5, stride=1, dir=args.folder, stage='test', shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for index_list, batch in test_loader:
            inputs = torch.tensor([], device=device)
            preds = torch.tensor([], device=device)
            X = batch
            y = batch

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
                all_latents = torch.cat([inputs[:,:-1], preds], dim=1)
                X = all_latents[:, -5:] # the next input is the last 5 frames of the concatenated inputs and preds
                print('X after modifying: ', X.shape)

            if args.show:
                for latent in all_latents.squeeze(0):
                    latent = latent.reshape((1, 4, 8, 8))
                    img = sd_utils.decode_img_latents(latent)
                    # img_path = os.path.join('./images', str(folder_index), str(index_list[idx - 1].item()) + '_gt.png')
                    # input_img[0].save(img_path)
                    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('frame', np.array(img[0]))
                    cv2.waitKey(0)


    #     # counting number of files in ./checkpoints
    #     folder_index = len(os.listdir('./images'))   
    #     os.mkdir('./images/' + str(folder_index))
    #     img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')
    #     pred_img[0].save(img_path)
    #     # pred_img[0].show()