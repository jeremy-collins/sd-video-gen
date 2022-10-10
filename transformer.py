import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
import math
import numpy as np

class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens=0,
        dim_model=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.height = 8
        self.width = 8

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=6
        )
        # self.embedding = nn.Embedding(num_tokens, dim_model)
        self.embedding = nn.Linear(self.height * self.width * 4, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, self.height * self.width * 4)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        
        # Src size must be (batch_size, src sequence length, dim_model)
        # Tgt size must be (batch_size, tgt sequence length, dim_model)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        # out = transformer_out
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        # mask = torch.zeros(size, size)
        # mask[-1] = 1
        # mask[]
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # mask = self.transformer.generate_square_subsequent_mask(1)
        
        # EX for size=5:
        # [[0.,   0.,   0.,   0.,   -inf.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)