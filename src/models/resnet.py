import torch
import torch.nn as nn
import torchvision.models as models

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()
        # Compress channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # spatial features - apply stride
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # expand channels back out
        self.conv3  = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward (self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)

        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 2):
        super().__init__()
        self.in_channels = 64

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # only first block strides
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
if __name__ == "__main__":
    model = ResNet(BottleneckBlock, [3, 4, 6, 3], 2)
    dummy = torch.randn(1, 3, 64, 64) # change 1 to 8 or something higher on better machine
    output = model(dummy)
    print ("Output shape", output.shape) # correct output - [1, 2]
