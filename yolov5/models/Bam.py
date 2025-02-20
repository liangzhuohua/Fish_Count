from torch import nn
from torch.nn import init
class Bam(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(Bam, self).__init__()

        # Compression layer in the bottleneck structure
        self.compress = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()

        # The attention mechanism in bottleneck structure
        self.conv_atten = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # Compressed portion
        x_compress = self.compress(x)  # Perform channel compression using 1x1 convolution.
        x_compress = self.relu(x_compress)  # ReLU activation function

        # Attention section
        atten = self.conv_atten(x_compress)  # Generate attention weights by using 1x1 convolution.
        atten = self.sigmoid(atten)  # Limit the attention weights within the range of 0 to 1 by using the Sigmoid function.

        # The final output is obtained by weighted fusion of the input features.
        out = x * atten  # Weighted fusion of feature maps is carried out by utilizing attention weights.
        return out
