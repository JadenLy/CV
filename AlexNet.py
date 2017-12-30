import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
This is an implementation of AlexNet on IMAGENET with PyTorch defined in

Alex Krizhevsky Ilya Sutskever Geoffrey E. Hinton (2012
ImageNet Classification with Deep Convolutional Neural Networks
"""


# Parameters
lr = 0.1
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
    model = AlexNet()
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

# Create Network Model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classify = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classify(x)

        return x

# Train and test the model
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

# Adjust the learning rate of optimizer by a factor of gamma
def adjust_lr(optimizer, gamma):
    for lr in optimizer.param_groups:
        lr['lr'] /= gamma
