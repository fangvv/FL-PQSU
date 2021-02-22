import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optimizer
from VGG import vgg
import random
import datetime
import numpy as np
import Quantification as Quan

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
loss_data = []
all_loss_data = []
list_name = []
save_scale=[]
save_zero_point=[]
load_scale=[]
load_zero_point=[]

def getName():
    for name in net.state_dict():
        list_name.append(name)

#加载未量化模型
def load_model():
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveModel/AverModel/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        net.state_dict()[var_name].copy_(tensor_load)


#存储非量化模型
def save_model():
    for name in list_name:
        temp_np = net.state_dict()[name].cpu().numpy()
        np.save("./SaveModel/SaveEdgeOneModel/%s.ndim" % (name), temp_np)


#存储量化模型
def quan_save_model():
    for varName in list_name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(net.state_dict()[varName])
        net.state_dict()[varName].copy_(temp_tensor)
        temp_list = net.state_dict()[varName].type(dtype=torch.uint8).cpu().numpy()
        np.save("./SaveModel/Quan_SaveEdgeModel/%s.ndim.npy" % (varName), temp_list)
        save_scale.append(temp_scale.item())
        save_zero_point.append(temp_zeropoint)

    with open("./SaveModel/Quan_SaveEdgeModel/scale.txt", 'w') as f:
        for temp_np in save_scale:
            f.write(str(temp_np) + "Q")
        f.close()

    with open("./SaveModel/Quan_SaveEdgeModel/zero_point.txt", 'w') as f:
        for temp_zero in save_zero_point:
            f.write(str(temp_zero) + "Q")
        f.close()


#加载量化模型
def quan_load_model():
    with open("./SaveModel/Quan_AverModel/scale.txt", 'r') as f:
        temp_scale_str = f.read()
        temp_scale = temp_scale_str.split("Q")
        for i in range(len(temp_scale) - 1):
            trans_scale_np = np.array(temp_scale[i], dtype=float)
            trans_scale_tensor = torch.tensor(trans_scale_np)
            load_scale.append(trans_scale_tensor)
        f.close()
    with open("./SaveModel/Quan_AverModel/zero_point.txt", 'r') as f:
        temp_zero_str = f.read()
        temp_zero = temp_zero_str.split("Q")
        for i in range(len(temp_zero) - 1):
            load_zero_point.append(int(temp_zero[i]))
    index = 0
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveModel/Quan_AverModel/%s.ndim.npy" % (var_name))
        tensor_load_numpy = torch.tensor(temp_load_numpy)
        test_para = Quan.dequantize_tensor(load_scale[index], tensor_load_numpy, load_zero_point[index])
        net.state_dict()[var_name].copy_(test_para)
        index += 1

#划分训练数据  
def get_train_data(index):
    train_x = []
    train_y = []
    for i in range(index, index + 200, 1):
        train_x.append(trainSet.__getitem__(i)[0].numpy())
        train_y.append(trainSet.__getitem__(i)[1])
    train_x = torch.tensor(train_x).view(-1, 3, 32, 32)
    train_y = torch.tensor(train_y)
    return train_x, train_y

#训练函数
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
            all_loss_data.append(loss.item())
            print("第" + str(i) + "次epoch 第" + str(j)+"次LossValue：" + str(loss.item()))
        loss_data.append(temp_loss / len(numbers))
        print("第%d次epoch，LossValue为%0.3f" % (i, temp_loss / len(numbers)))


def saveLoss():
    with open("./SaveLoss/saveLoss_Edge_pruning_quan_imp.txt", 'a') as fwirte:
        for i in range(len(loss_data)):
            fwirte.write(str(loss_data[i]) + "-")
    fwirte.close()


def saveAllLoss():
    with open("./SaveAllLoss/saveAllLoss_Edge_pruning_quan_imp.txt", 'a') as fwirte:
        for i in range(len(all_loss_data)):
            fwirte.write(str(all_loss_data[i]) + "-")
    fwirte.close()


def saveTime(time):
    with open("./SaveTime/saveTime_EdgeOne_pruning_quan_imp.txt", 'a') as fwtime:
        fwtime.write(str(float(time)) + '-')
    fwtime.close()

def Trans_Value():
    compare_loss=all_loss_data[-1]
    with open("./SaveTransInfo/No_Data/EdgeOne_Loss.txt",'r') as readValue:
        temp_str=readValue.read()
    readValue.close()
    Loss_value=temp_str.split('-')
    Loss_value=Loss_value[:len(Loss_value)-1]
    if len(Loss_value)==0:
        Last_loss=100
    else:
        Last_loss=Loss_value[-1]
    print("Last_loss："+str(Last_loss)+"   compare_loss："+str(compare_loss))
    if compare_loss<float(Last_loss):
        with open("./SaveTransInfo/No_Data/EdgeOne_Loss.txt",'a') as writeValue:
            writeValue.write(str(compare_loss)+"-")
        writeValue.close()
        Save_Sign(1)
        return 1
    else:
        Save_Sign(0)
        return 0

def Save_Sign(sign):
    with open("./SaveTransInfo/No_Data/EdgeOne_Sign.txt",'a') as writeSign:
        if sign==1:
            writeSign.write("T-")
        else:
            writeSign.write("F-")
        writeSign.close()

if __name__ == "__main__":
    before_train = datetime.datetime.now()
    numbers = list(range(0, 20000, 200))
    getName()
    quan_load_model()
    train(6)
    saveLoss()
    saveAllLoss()
    if Trans_Value()==1:
        quan_save_model()
        print("传输数据")
    after_train = datetime.datetime.now()
    saveTime((after_train - before_train).total_seconds())
