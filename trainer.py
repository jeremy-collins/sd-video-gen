import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from transformer import Transformer
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from bouncing_ball_loader import BouncingBall
import argparse
import os

def train_loop(model, opt, loss_fn, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    model = model.to(device)
    model.train()
    total_loss = 0
    for i, (index_list, batch) in enumerate(dataloader):
        print('batch', i)
        if i >= 100:
            break
        # X, y = batch[:, 0], batch[:, 1]
        
        # X = batch[:, :-1]
        # y = batch[:,-1].unsqueeze(1)
        
        X = batch
        y = batch
        
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
        
        y_input = y
        y_expected = y
        
        y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
        y_expected = y_expected.permute(1, 0, 2)
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)
        # pred = None

        # Permute pred to have batch size first again
        # pred = pred.permute(1, 2, 0)
        
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    # return total_loss / len(dataloader)
    return total_loss / 100.0


def validation_loop(model, loss_fn, dataloader):  
    model.eval()
    total_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for j, (index_list, batch) in enumerate(dataloader):
            print('batch', j)
            if j >= 100:
                break
        
            # X, y = batch[:, 0], batch[:, 1]
            X = batch[:, :-1]
            y = batch
            
            # X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
            X, y = torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device) # TODO: check if this is correct

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            y_expected = y_expected.reshape(y_expected.shape[0], y_expected.shape[1], -1)
            y_expected = y_expected.permute(1, 0, 2)
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 0, 2)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)
    # return total_loss / 100.0

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs): 
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
    
    # counting number of files in ./checkpoints
    index = len(os.listdir('./checkpoints'))    
    
    # save model
    torch.save(model.state_dict(), './checkpoints/model' + '_' + str(index) + '.pt')
    print('model saved as model' + '_' + str(index) + '.pt')
        
    return train_loss_list, validation_loss_list

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    # torch.multiprocessing.set_start_method('spawn')
    
    model = Transformer(num_tokens=0, dim_model=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss() # TODO: change this to mse + condition + gradient difference
    
    frames_per_clip = 5
    step_between_clips = 1
    batch_size = 4
    
    if args.dataset == 'ucf101':    
        ucf_data_dir = "/Users/jsikka/Documents/UCF-101"
        ucf_label_dir = "/Users/jsikka/Documents/ucfTrainTestlist"
        

        tfs = transforms.Compose([
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                # rescale to the most common size
                transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
        ])

        train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                        step_between_clips=step_between_clips, train=True, transform=tfs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        # create test loader (allowing batches and other extras)
        test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                            step_between_clips=step_between_clips, train=False, transform=tfs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        
    elif args.dataset == 'ball':
        train_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='train', shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = BouncingBall(num_frames=5, stride=1, dir='/media/jer/data/bouncing_ball_1000_1/test1_bouncing_ball', stage='test', shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        

    # # print(train_loader)
    # print("TRAIN LOADER")
    # for i in train_loader:
    #     print(len(i))
    #     print(i.size())
    #     print(i)
    #     break

    # print("TEST LOADER")
    # # print(test_loader)
    # for i in test_loader:
    #     print(i.size())
    #     print(i)
    #     break

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_loader, test_loader, 1)