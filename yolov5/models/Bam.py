from torch import nn
from torch.nn import init
class Bam(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(Bam, self).__init__()

        # 瓶颈结构中的压缩层
        self.compress = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()

        # 瓶颈结构中的注意力机制
        self.conv_atten = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # 压缩部分
        x_compress = self.compress(x)  # 使用1x1卷积进行通道压缩
        x_compress = self.relu(x_compress)  # ReLU激活函数

        # 注意力部分
        atten = self.conv_atten(x_compress)  # 利用1x1卷积生成注意力权重
        atten = self.sigmoid(atten)  # 使用Sigmoid函数将注意力权重限制在0到1之间

        # 对输入特征加权融合得到最终输出
        out = x * atten  # 利用注意力权重对特征图进行加权融合
        return out