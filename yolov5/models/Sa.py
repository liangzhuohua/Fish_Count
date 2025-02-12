import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class Sa(nn.Module):
    def __init__(self, in_channels):
        super(Sa, self).__init__()

        # 初始化查询、键、值的卷积层
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 初始化注意力权重调节参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # 计算查询、键、值
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # 计算注意力能量和注意力权重
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # 计算自注意力向量
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)

        # 将自注意力向量与原始特征相加并乘以注意力权重调节参数
        out = self.gamma * out + x
        return out