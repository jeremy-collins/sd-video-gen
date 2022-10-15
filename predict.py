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

def predict(model, input_sequence, max_length=5, SOS_token=2, EOS_token=3):
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
        for _ in range(6):
        # for _ in range(max_length):
            # Get source mask
            print('input_sequence: ', input_sequence.shape)
            print('y_input: ', y_input.shape)
            
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            
            pred = model(input_sequence, y_input, tgt_mask)
            
            # Permute pred to have batch size first again
            pred = pred.permute(1, 0, 2)
            
            # next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            print('pred', pred.shape)
            
            # X shape is (batch_size, src sequence length, input.shape)
            # y_input shape is (batch_size, tgt sequence length, input.shape)
        
            # next item is the last item in the predicted sequence
            next_item = pred[:,-1,:].unsqueeze(1)
            # next_item = torch.tensor([[next_item]], device=device)
            
            print('next_item: ', next_item.shape)

            # Concatenate previous input with prediction
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            # if next_item.view(-1).item() == EOS_token:
            #     break

    # return y_input.view(-1).tolist()
    # return pred[0,0].view(-1).tolist()
    return pred[0, -1]
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    args = parser.parse_args()
    
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.load_state_dict(torch.load('./checkpoints/model_' + str(args.index) + '.pt'))
    model.eval()
    model = model.to(device)
    
    # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='test', shuffle=True)
    # test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_blackwhite1/content/2D-bouncing/test3_bouncing_ball', stage='test', shuffle=True)
    test_dataset = BouncingBall(num_frames=5, stride=3, dir='/media/jer/data/tccvg/bouncing_ball_3000_blackwhite_simple1/content/2D-bouncing/test2_simple_bouncing_ball', stage='test', shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for index_list, batch in test_loader:
            X = batch
            y = batch
            
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)

            # # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            # y_expected = y[:,1:]
            
            # y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
            # y_expected = y_expected.permute(1, 0, 2)
            
            # # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            
            # X shape is (batch_size, src sequence length, input.shape)
            result = predict(model, X)
            result = torch.tensor(result, dtype=torch.float32, device=device)

            pred_latents = result.reshape((1, 4, 8, 8))
            pred_img = sd_utils.decode_img_latents(pred_latents)
            
            # counting number of files in ./checkpoints
            folder_index = len(os.listdir('./images'))   
            os.mkdir('./images/' + str(folder_index))
            img_path = os.path.join('./images', str(folder_index), str(index_list[-1].item()) + '_pred.png')

            pred_img[0].save(img_path)
            
            # pred_img[0].show()
            
            for idx, input in enumerate(X.squeeze(0)):
                if idx == 0:
                    continue # SOS token
                print('X', X.shape)
                input_latents = input.reshape((1, 4, 8, 8))
                input_img = sd_utils.decode_img_latents(input_latents)
                img_path = os.path.join('./images', str(folder_index), str(index_list[idx - 1].item()) + '_gt.png')
                input_img[0].save(img_path)
            
            # print('result', result)
            # print('result', result.shape)
            # print('pred_latents', pred_latents.shape)
            # print('pred_img', pred_img[0].shape)

            # # Standard training except we pass in y_input and src_mask
            # pred = model(X, y_input, tgt_mask)

            # # Permute pred to have batch size first again
            # # (batch_size, sequence_length, num_latents)
            # pred = pred.permute(1, 0, 2)
