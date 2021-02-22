import sys
sys.path.append("/mnt/FLVGG/VGG")
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optimizer
from VGG import vgg
import random
import datetime
import numpy as np

prunig_lay=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
net = vgg(depth=16, cfg=prunig_lay,fullLayers=[4096, 4096])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.cuda()

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainSet = torchvision.datasets.CIFAR10(root='./CifarTrainData', train=True,
                                        download=True, transform=transform)

loss_fc = nn.CrossEntropyLoss()
optimizer = optimizer.SGD(net.parameters(), lr=0.1, momentum=0.9)
list_name = []

def getName():
    for name in net.state_dict():
        list_name.append(name)


def loadModel():
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveOrgModel/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        net.state_dict()[var_name].copy_(tensor_load)


def saveModel():
    for name in list_name:
        temp_np = net.state_dict()[name].cpu().numpy()
        np.save("./SaveOrgModel/%s.ndim" % (name), temp_np)


def get_train_data(index):
    train_x = []
    train_y = []
    for i in range(index, index + 200, 1):
        train_x.append(trainSet.__getitem__(i)[0].numpy())
        train_y.append(trainSet.__getitem__(i)[1])
    train_x = torch.tensor(train_x).view(-1, 3, 32, 32)
    train_y = torch.tensor(train_y)
    return train_x, train_y


def train(epoch):
    for i in range(epoch):
        temp_loss = 0
        random.shuffle(numbers)
        for j in range(len(numbers)):
            train_x, train_y = get_train_data(numbers[j])
            train_x, train_y = train_x.to(device), train_y.to(device)
            prediction = net.forward(train_x)
            loss = loss_fc(prediction, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            print("第" + str(i) + "次epoch 第" + str(j) + "次LossValue：" + str(loss.item()))
        print("第%d次epoch，LossValue为%0.3f" % (i, temp_loss / len(numbers)))


if __name__ == "__main__":
    before_train = datetime.datetime.now()
    numbers = list(range(15000, 20000, 200))
    numbers1=list(range(35000,40000,200))
    for i in range(len(numbers1)):
        numbers.append(numbers1[i])
    print(len(numbers))
    getName()
    loadModel()
    train(6)
    saveModel()
