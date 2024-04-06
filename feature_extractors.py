import math
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm1d(outplanes),
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResNet1D, self).__init__()

        self.inplanes = in_dim
        self.downsample_block = downsample_basic_block

        # add an extra conv layer. Note that this layer is not used in the original ResNet
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, self.inplanes, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), (1, 2), (0, 0)),  # perform 1/2 downsampling. seq_len, 64, 80 -> seq_len, 64, 40
        )

        layers = [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, out_dim, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(out_dim)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(inplanes=self.inplanes,
                                               outplanes=planes * block.expansion,
                                               stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_lens):
        x = x.unsqueeze(1)          # (b, 1, t, 80)
        x = self.conv2d(x)          # (b, 64, t, 40)
        x = x.permute(0, 2, 1, 3)   # (b, t, 64, 40)

        b, t, c, l = x.size()
        x = x.contiguous().view(b * t, c, l)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         # (b*t, out_dim, 1)
        x = x.view(b * t, -1)       # (b*t, out_dim)

        x = self.bn(x)
        x = x.view(b, t, -1)        # (b, t, out_dim)

        # perform subsampling by stacking every 4 time steps
        if x.size(1) % 4 != 0:
            x = x[:, :-(x.size(1) % 4), :]
        x = x.contiguous().view(b, -1, 4, x.size(2)).mean(2)  # (b, t//4, out_dim)
        x_lens = x_lens // 4

        return x, x_lens


class LinearFeatureExtractionModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x, x_lens):
        return self.linear(x), x_lens


if __name__ == "__main__":
    model = ResNet1D(64, 256)

    x = torch.randn(2, 100, 80)
    x_lens = torch.tensor([100])

    out, out_lens = model(x, x_lens)
    print(x.size(), out.size(), x_lens, out_lens)
