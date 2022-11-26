import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
import math
import numpy as np
from config import parse_config_args
from sentence_transformers import SentenceTransformer

def load_st_weights():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model#.to(device)

class Transformer(nn.Module):
        # setting self.learned_tgt to a parameter so that it can be optimized
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

        self.config, self.args = parse_config_args()

        # TODO: If we want to use 2048 + text_embed_dim, comment line #32 and uncomment line #34
        # self.dim_model = dim_model
        self.text_embed_dim = 384
        self.dim_model = dim_model + self.text_embed_dim
        self.img_embed_dim = self.dim_model - self.text_embed_dim # ultimately img_embed_dim is equal to config.dim_model

        # IMAGE SIZE
        self.height = self.config.FRAME_SIZE
        self.width = self.config.FRAME_SIZE
        self.compression = 8

        print("loading st_weights")
        # Sentence Transformer weights
        self.sent_transformer = load_st_weights()
        print("loaded st_weights")

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=self.dim_model, dropout_p=dropout_p, max_len=64
        )
        # self.embedding = nn.Embedding(num_tokens, dim_model)
        # self.embedding = nn.Linear(self.height // self.compression * self.width // self.compression * 4, dim_model)
        print("defining image embedding")
        #print(self.dim_model)
        #print(self.text_embed_dim)
        #print(type(self.dim_model))
        #print(type(self.text_embed_dim))
        #print(self.img_embed_dim)
        #print(type(self.img_embed_dim))
        self.project_image_embedding = nn.Linear(self.height // self.compression * self.width // self.compression * 4, self.img_embed_dim)
        print("defined image embedding")
        self.transformer = nn.Transformer(
            d_model=self.dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(self.dim_model, self.height // self.compression  * self.width // self.compression * 4)
        
    def forward(self, src, cls_list, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        
        # Src size must be (batch_size, src sequence length, dim_model)
        # Tgt size must be (batch_size, tgt sequence length, dim_model)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # TODO: convert to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, T_src, _ = src.shape
        _, T_tgt, _ = tgt.shape
        txt = self.sent_transformer.encode(cls_list)
        txt = torch.from_numpy(txt).to(device)
        txt_src = txt.unsqueeze(1).repeat(1,T_src,1)
        txt_tgt = txt.unsqueeze(1).repeat(1,T_tgt,1)
        #print("txt_src.shape", txt_src.shape) #batch_size x T x 384
        #print("txt_tgt.shape", txt_tgt.shape) #batch_size x T x 384

        src = self.project_image_embedding(src) #batch_size x 1664 (1664+384=2048=dim_model)
        src = torch.cat((src, txt_src), dim=-1) * math.sqrt(self.dim_model)
        tgt = self.project_image_embedding(tgt)
        tgt = torch.cat((tgt, txt_tgt), dim=-1) * math.sqrt(self.dim_model)

        # src = self.embedding(src) * math.sqrt(self.dim_model)
        # tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        #print("src.shape:", src.shape)
        #print("tgt.shape:", tgt.shape)
        
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
