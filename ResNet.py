import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math

"""
This is an implementation of ResNet-50 on IMAGENET with PyTorch as defined in

Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun (2015)
Deep Residual Learning for Image Recognition
arXiv preprint arXiv:1512.03385
"""

# Parameter
lr = 0.01
momentum = 0.9
batch_size = 32
weight_decay = 1e-4
trainDirectory = ""
valDirectory = ""
gamma = 0.1
iter_step = 100000
max_iter = 1000000
cuda = False and torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    kwargs = {}

def main():
    trainLoad, testLoad = prepare(batch_size, trainDirectory, valDirectory, kwargs)
    model = ResNet([3, 4, 6, 3])
    train_test(model, lr, momentum, weight_decay, trainLoad, testLoad, gamma, max_iter, iter_step)

# Load data
def prepare(batch_size, trainDirectory, valDirectory, kwargs):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    transforms.RandomSizedCrop(224),
                                    transforms.RandomHorizontalFlip()])

    trainData = torchvision.datasets.ImageFolder(trainDirectory, transform=transform)
    trainLoad = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, **kwargs)
    testData = torchvision.datasets.ImageFolder(valDirectory, transform=transform)
    testLoad = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, **kwargs)

    return trainLoad, testLoad


# Create Network
class ResNet(nn.Module):
    def __init__(self, layers):
        self.input = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_block(64, layers[0])
        self.block2 = self._make_block(128, layers[1], 2)
        self.block3 = self._make_block(256, layers[2], 2)
        self.block4 = self._make_block(512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, output, layer, stride=1):
        downsample = None
        if stride != 1 or self.input != output*4:
            downsample = nn.Sequential(nn.Conv2d(self.input, output*4, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(output*4))

        layers = []
        layers.append(BottleNeck(self.input, output, stride, downsample))
        self.input = output*4
        for i in range(1, layer):
            layers.append(BottleNeck(self.input, output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Create the Block structure in ResNet
class BottleNeck(nn.Module):
    def __init__(self, input, output, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output)
        self.conv3 = nn.Conv2d(output, output*4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(output*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU(inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample != None:
            residual = self.downsample(x)

        out += residual
        out = nn.ReLU(inplace=True)

        return out

# Train the model and test with test data
def train_test(model, lr, momentum, weight_decay, trainLoad, testLoad, gamma, max_iter, iter_step):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    iter = 1

    while iter < max_iter:

        if iter % iter_step == 0:
            optimizer = adjust_lr(optimizer, gamma)

        for i, data in enumerate(trainLoad):
            input, label = data
            input, label = Variable(input), Variable(label)

            optimizer.zero_grad()
            output = model(input)
            train_loss = F.nll_loss(output, label)
            train_loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                test_loss = 0
                right = 0

                for data, target in testLoad:
                    data, target = Variable(data), Variable(target)
                    result = model(data)
                    test_loss += F.nll_loss(result, target, size_average=False),data[0]
                    pred = result.data.max(1, keepdim=True)[1]
                    right += pred.eq(target.data.view_as(pred)).sum()

                print("Iteration {}: Training Loss: {}, Test Loss: {}, Test Accuracy: {}%".format(iter,
                        train_loss.data.numpy()[0], test_loss, 100. * right / len(testLoad.dataset())))
            iter += 1


def adjust_lr(optimizer, gamma):
    for lr in optimizer.param_groups:
        lr['lr'] /= gamma


