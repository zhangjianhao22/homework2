# cifar10_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ----------------------------
# 1. 模型定义
# ----------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 64

        # CIFAR-10 图像小（32x32），所以第一层用较小卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]

        x = self.layer1(x)  # [B, 64, 32, 32]
        x = self.layer2(x)  # [B, 128, 16, 16]
        x = self.layer3(x)  # [B, 256, 8, 8]
        x = self.layer4(x)  # [B, 512, 4, 4]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18_cifar():
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2])


# ----------------------------
# 2. 数据加载与增强
# ----------------------------
def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


# ----------------------------
# 3. 训练函数
# ----------------------------
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return running_loss / len(trainloader), acc


# ----------------------------
# 4. 测试函数
# ----------------------------
def test(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


# ----------------------------
# 5. 主训练循环
# ----------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 超参数
    epochs = 100
    batch_size = 128
    learning_rate = 0.1

    # 数据
    trainloader, testloader = get_dataloaders(batch_size)

    # 模型
    model = resnet18_cifar().to(device)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)

    # 学习率调度：每 60 轮衰减为 0.1 倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        test_acc = test(model, testloader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_resnet_cifar10.pth')

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)")

    print(f"\n✅ Training finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()