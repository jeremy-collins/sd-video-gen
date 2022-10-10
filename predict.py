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

def predict(model, input_sequence, max_length=5, SOS_token=2, EOS_token=3):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
    y_input = torch.tensor(torch.zeros((1,1,256)), dtype=torch.float32, device=device)

    # num_tokens = len(input_sequence[0])
    with torch.no_grad():
        for _ in range(1):
        # for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            
            pred = model(input_sequence, y_input, tgt_mask)
            
            # next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            print('pred', pred.shape)
            next_item = pred[:,-1,:].unsqueeze(1)
            # next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            # if next_item.view(-1).item() == EOS_token:
            #     break

    # return y_input.view(-1).tolist()
    return pred[0,0].view(-1).tolist()
    
  
if __name__ == "__main__":  
    sd_utils = SDUtils()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.load_state_dict(torch.load('./checkpoints/model_0.pt'))
    model.eval()
    model = model.to(device)
    
    test_dataset = BouncingBall(num_frames=5, fps=30, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='test', shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X = batch[:, :-1]
            y = batch
            
            X, y = torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)

            # # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            # y_expected = y[:,1:]
            
            # y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
            # y_expected = y_expected.permute(1, 0, 2)
            
            # # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            
            result = predict(model, X)
            result = np.array(result)
            pred_latents = result.reshape((1, 4, 8, 8))
            pred_latents = torch.tensor(pred_latents, dtype=torch.float32, device=device)
            pred_img = sd_utils.decode_img_latents(pred_latents)
            
            # pred_img = pred_img[0].cpu().numpy()
            
            # cv2.imshow('pred', pred_img)
            pred_img[0].show()
            
            # print('result', result)
            # print('result', result.shape)
            # print('pred_latents', pred_latents.shape)
            # print('pred_img', pred_img[0].shape)

            # # Standard training except we pass in y_input and src_mask
            # pred = model(X, y_input, tgt_mask)

            # # Permute pred to have batch size first again
            # # (batch_size, sequence_length, num_latents)
            # pred = pred.permute(1, 0, 2)
        
        
        # # Here we test some examples to observe how the model predicts
        # examples = [
        #     torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
        #     torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        #     torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        #     torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
        #     torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        #     torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
        # ]

        # for idx, example in enumerate(examples):
        #     result = predict(model, example)
        #     print(f"Example {idx}")
        #     print(f"Input: {example.view(-1).tolist()[1:-1]}")
        #     print(f"Continuation: {result[1:-1]}")
        #     print()