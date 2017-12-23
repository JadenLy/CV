import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# This is an implementation of Inception-v3 on IMAGENET with PyTorch


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
    model = Inception()
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

# Create the Network
class Inception(nn.Module):

    def __init__(self, aux_logits=True, transform_input=False):
        super(Inception, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2)
        self.conv2 = ConvBn(32, 32, kernel_size=3)
        self.conv3 = ConvBn(32, 64, kernel_size=3, padding=1)
        self.conv4 = ConvBn(64, 80, kernel_size=1)
        self.conv5 = ConvBn(80, 192, kernel_size=3)

        self.conv6a = BlockA(192, 32)
        self.conv6b = BlockA(256, 64)
        self.conv6c = BlockA(288, 64)

        self.conv7a = BlockB(288)
        self.conv7b = BlockC(768, 128)
        self.conv7c = BlockC(768, 160)
        self.conv7d = BlockC(768, 160)
        self.conv7e = BlockC(768, 192)

        if aux_logits:
            self.aux_logits = BlockAux(768)

        self.conv8a = BlockD(768)
        self.conv8b = BlockE(1280)
        self.conv8c = BlockE(2048)
        self.fc = nn.Linear(2048, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv6a(x)
        x = self.conv6b(x)
        x = self.conv6c(x)
        x = self.conv7a(x)
        x = self.conv7b(x)
        x = self.conv7c(x)
        x = self.conv7d(x)
        x = self.conv7e(x)

        if self.training and self.aux_logits:
            aux = self.aux_logits(x)

        x = self.conv8a(x)
        x = self.conv8b(x)
        x = self.conv8c(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux
        return x


class BlockA(nn.Module):

    def __init__(self, input, features):
        super(BlockA, self).__init__()
        self.conv1 = ConvBn(input, 64, kernel_size=1)
        self.conv3a = ConvBn(64, 96, kernel_size=3, padding=1)
        self.conv3b = ConvBn(96, 96, kernel_size=3, padding=1)
        self.conv5a = ConvBn(input, 48, kernel_size=1)
        self.conv5b = ConvBn(48, 64, kernel_size=5, padding=2)
        self.pool = ConvBn(input, features, kernel_size=1)

    def forward(self, x):
        a = self.con1(x)

        b = self.conv5a(x)
        b = self.conv5b(b)

        c = self.conv1(x)
        c = self.conv3a(c)
        c = self.conv3b(c)

        d = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        d = self.pool(d)

        return torch.cat([a, b, c, d], 1)


class BlockB(nn.Module):

    def __init__(self, input):
        super(BlockB, self).__init__()
        self.conv1 = ConvBn(input, 384, kernel_size=3, stride=2)
        self.conv3a = ConvBn(input, 64, kernel_size=1)
        self.conv3b = ConvBn(64, 96, kernel_size=3, padding=1)
        self.conv3c = ConvBn(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        a = self.conv1(x)

        b = self.conv3a(x)
        b = self.conv3b(b)
        b = self.conv3c(b)

        c = F.max_pool2d(x, kernel_size=3, stride=2)

        return torch.cat([a, b, c], 1)


class BlockC(nn.Module):

    def __init__(self, input, channels):
        super(BlockC, self).__init__()

        self.conv1 = ConvBn(input, 192, kernel_size=1)

        self.conv7a = ConvBn(input, channels, kernel_size=1)
        self.conv7b = ConvBn(channels, channels, kernel_size=(1, 7), padding=(0, 3))
        self.conv7c = ConvBn(channels, 192, kernel_size=(7, 1), padding=(3, 0))

        self.conv17a = ConvBn(input, channels, kernel_size=1)
        self.conv17b = ConvBn(channels, channels, kernel_size=(7, 1), padding=(3, 0))
        self.conv17c = ConvBn(channels, channels, kernel_size=(1, 7), padding=(0, 3))
        self.conv17d = ConvBn(channels, channels, kernel_size=(7, 1), padding=(3, 0))
        self.conv17e = ConvBn(channels, 192, kernel_size=(1, 7), padding=(0, 3))

        self.pool = ConvBn(input, 192, kernel_size=1)

    def forward(self, x):
        a = self.conv1(x)

        b = self.conv7a(x)
        b = self.conv7b(b)
        b = self.conv7c(b)

        c = self.conv17a(x)
        c = self.conv17b(c)
        c = self.conv17c(c)
        c = self.conv17d(c)
        c = self.conv17e(c)

        d = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        d = self.pool(d)

        return torch.cat([a, b, c, d], 1)


class BlockD(nn.Module):

    def __init__(self, input):
        super(BlockD, self).__init__()
        self.conv3a = ConvBn(input, 192, kernel_size=1)
        self.conv3b = ConvBn(192, 320, kernel_size=3, stride=2)

        self.conv7a = ConvBn(input, 192, kernel_size=1)
        self.conv7b = ConvBn(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.conv7c = ConvBn(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.conv7d = ConvBn(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        a = self.conv3a(x)
        a = self.conv3b(a)

        b = self.conv7a(x)
        b = self.conv7b(b)
        b = self.conv7c(b)
        b = self.conv7d(b)

        c = F.max_pool2d(x, kernel_size=3, stride=2)

        return torch.cat([a, b, c], 1)


class BlockE(nn.Module):

    def __init__(self, input):
        super(BlockE, self).__init__()
        self.conv1 = ConvBn(input, 320, kernel_size=1)

        self.conv3a = ConvBn(input, 384, kernel_size=1)
        self.conv3b = ConvBn(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.conv3c = ConvBn(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.conv13a = ConvBn(input, 448, kernel_size=1)
        self.conv13b = ConvBn(448, 384, kernel_size=3, padding=1)
        self.conv13c = ConvBn(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.conv13d = ConvBn(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.pool = ConvBn(input, 192, kernel_size=1)

    def forward(self, x):
        a = self.conv1(x)

        b = self.conv3a(x)
        b = [self.conv3b(b), self.conv3c(b)]
        b = torch.cat(b, 1)

        c = self.conv13a(x)
        c = self.conv13b(c)
        c = [self.conv13c(c), self.conv13d(c)]
        c = torch.cat(c, 1)

        d = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        d = self.branch_pool(d)

        return torch.cat([a, b, c, d], 1)


class BlockAux(nn.Module):

    def __init__(self, input):
        super(BlockAux, self).__init__()
        self.conv0 = ConvBn(input, 128, kernel_size=1)
        self.conv1 = ConvBn(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, 1000)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvBn(nn.Module):
    def __init__(self, input, output, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(input, output, **kwargs)
        self.bn = nn.BatchNorm2d(output, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


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





