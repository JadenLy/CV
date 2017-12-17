import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# This is an implementation of LeNet on MNIST with PyTorch


# Parameters
batch_size = 64
lr = 0.01
momentum = 0.5
num_epoch = 10
cuda = False and torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    kwargs = {}

# Dataset Process

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trainData = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainLoad = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, **kwargs)
testData = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testLoad = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, **kwargs)


# Network Structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)


model = Net()

# Optimization
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# train the network and test the model after each
def train_test():
    model.train()
    iter = 0

    for epoch in range(num_epoch):

        for i, data in enumerate(trainLoad):
            input, label = data
            input, label = Variable(input), Variable(label)
            optimizer.zero_grad()
            output = model(input)
            train_loss = F.nll_loss(output, label)
            train_loss.backward()
            optimizer.step()
			
			# Test on the test set each 100 iterations. 						
            if iter % 100 == 0:
                test_loss = 0
                right = 0

                for data, target in testLoad:
                    data, target = Variable(data), Variable(target)
                    result = model(data)
                    test_loss += F.nll_loss(result, target, size_average=False).data[0]
                    pred = result.data.max(1, keepdim=True)[1]
                    right += pred.eq(target.data.view_as(pred)).sum()
				
				# print out the result				
                print("Iteration {}: Training Loss = {}, Testing Loss = {}, Test Accuracy = {}%".format(iter,
                                                train_loss.data.numpy()[0], test_loss, 100. * right / len(testLoad.dataset)))

            iter += 1


# Begin Training
train_test()

