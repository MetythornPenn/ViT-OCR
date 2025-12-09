import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        
        self.d_model = d_model # Dimension of the model
        self.img_size = img_size # Size (H, W)
        self.patch_size = patch_size # (Ph, Pw)
        self.n_channels = n_channels #  3 for RGB, 1 for Grayscale
        
        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return x

