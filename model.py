import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # We don't use functional relu inside the block right before adding,
        # but apply it after the shortcut for better AMP stability.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BloodMNISTResNet(nn.Module):
    """
    A lightweight ResNet architecture explicitly optimized for 28x28 images.
    Standard ResNet implementations (like torchvision.models.resnet18) are built 
    for 224x224 images and apply aggressive pooling early on, which would destroy 
    information on small 28x28 BloodMNIST images.
    """
    def __init__(self, num_classes=8):
        super(BloodMNISTResNet, self).__init__()
        self.in_planes = 64

        # Notice: kernel_size=3, stride=1, padding=1 instead of a huge 7x7 kernel with stride 2.
        # This preserves the spatial dimensions of the 28x28 input.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4 ResNet Layers
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)    # Output: 28x28
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)   # Output: 14x14
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)   # Output: 7x7
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)   # Output: 4x4
        
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    # Quick sanity check for the model footprint
    model = BloodMNISTResNet(num_classes=8)
    # Validate it works on a dummy batch of 128 images of 28x28 RGB
    dummy_input = torch.randn(128, 3, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 128, 8)")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
