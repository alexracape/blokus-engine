"""This architecture is based on the ViT"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlokusTransformer(nn.Module):
    def __init__(self, 
                 d_max=20,      # maximum board dimension
                 embed_dim=128, # dimension of the transformer embeddings
                 num_heads=4,
                 mlp_dim=256,   # dimension of feedforward layer in TransformerEncoderLayer
                 num_layers=4,  # number of transformer layers
                 dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(5, embed_dim)
        self.pos_embed_2d = nn.Parameter(torch.zeros(d_max, d_max, embed_dim))
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=mlp_dim,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(embed_dim, 4)
        
    def forward(self, board_tensor):
        """
        board_tensor: shape [batch_size, d, d, input_dim]
                      each cell has input_dim=5 features (4 occupancy bits + 1 legal bit)
        
        Returns:
            policy_logits: [batch_size, d, d]
            value:         [batch_size, 4]
        """
        batch_size, d, _, _ = board_tensor.shape

        # Flatten board cells: [batch_size, d*d, input_dim]
        x = board_tensor.view(batch_size, d*d, -1)
        
        # Project to embedding dim: [batch_size, d*d, embed_dim]
        x = self.input_proj(x)
        
        # Add positional embeddings
        pos_embed_slice = self.pos_embed_2d[:d, :d, :].reshape(d*d, -1)  # [d*d, embed_dim]
        pos_embed_slice = pos_embed_slice.unsqueeze(0)  # [1, d*d, embed_dim]
        x = x + pos_embed_slice

        init_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([init_cls_tokens, x], dim=1)
        
        x = self.transformer(x)  # [batch_size, 1 + d*d, embed_dim]
        board_tokens = x[:, 1:, :]
        cls_tokens = x[:, 0, :]

        policy_logits = self.policy_head(board_tokens).view(batch_size, d*d) # [batch_size, d * d]
        value_logits = self.value_head(cls_tokens)  # [batch_size, 4]
        
        return policy_logits, value_logits


if __name__ == "__main__":
    # Quick test
    batch_size = 2
    d = 8
    test_input = torch.randn(batch_size, d, d, 5)  # e.g. random board states
    model = BlokusTransformer(d_max=20, 
                              input_dim=5, 
                              embed_dim=64, 
                              num_heads=4, 
                              mlp_dim=128, 
                              num_layers=2)
    
    policy, value = model(test_input)
    print(policy.shape)  # expect [2, 8, 8]
    print(value.shape)   # expect [2, 1]






