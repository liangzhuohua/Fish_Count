import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class Sa(nn.Module):
    def __init__(self, in_channels):
        super(Sa, self).__init__()

        # Initialization query, key, and value convolutional layer
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Initialize the parameters for adjusting the attention weights.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # Calculate queries, keys, and values
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # Calculate attention energy and attention weights
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # Calculate the self-attention vector
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)

        # Add the self-attention vector to the original features and multiply them by the attention weight adjustment parameters.
        out = self.gamma * out + x
        return out
