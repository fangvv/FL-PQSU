import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import datetime
from AlexNet import AlexNet
import pandas as pd
import Quantification as Quan
import os

prunig_lay=[64,128,256,256,256,3300,3500]
train_data=pd.DataFrame(pd.read_csv("./Mnist_data/one.csv",encoding='utf-8'))
test_data=pd.DataFrame(pd.read_csv("./Mnist_data/mnist_test.csv"))
model=AlexNet(prunig_lay)

loss_fc = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
loss_list = []
list_Name = []

save_scale = []
save_zero_point = []
load_scale = []
load_zero_point = []


def getName():
    for list in model.state_dict():
        list_Name.append(list)


def train(epoch):
    for j in range(epoch):
        for i in range(10):
            batch_data = train_data.sample(n=30, replace=False)
            batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).long()
            batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
            prediction = model.forward(batch_x)
            loss = loss_fc(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("第%d次epoch第%d训练，loss为%0.3f" % (j, i, loss))
            loss_list.append(loss.item())


# 量化存储模型
def quan_save_model():
    for varName in list_Name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(model.state_dict()[varName])
        model.state_dict()[varName].copy_(temp_tensor)
        temp_list = model.state_dict()[varName].type(dtype=torch.uint8).numpy()
        np.save("../SaveModel/ImportQuanSaveModelEdge/%s.ndim.npy" % (varName), temp_list)
        save_scale.append(temp_scale.item())
        save_zero_point.append(temp_zeropoint)

    with open("../SaveModel/ImportQuanSaveModelEdge/scale.txt", 'w') as f:
        for temp_np in save_scale:
            f.write(str(temp_np) + "Q")
        f.close()

    with open("../SaveModel/ImportQuanSaveModelEdge/zero_point.txt", 'w') as f:
        for temp_zero in save_zero_point:
            f.write(str(temp_zero) + "Q")
        f.close()


# 加载量化存储模型
def quan_load_model():
    with open("../SaveModel/ImportAverModel/scale.txt", 'r') as f:
        temp_scale_str = f.read()
        temp_scale = temp_scale_str.split("Q")
        for i in range(len(temp_scale) - 1):
            trans_scale_np = np.array(temp_scale[i], dtype=float)
            trans_scale_tensor = torch.tensor(trans_scale_np)
            load_scale.append(trans_scale_tensor)
        f.close()
    with open("../SaveModel/ImportAverModel/zero_point.txt", 'r') as f:
        temp_zero_str = f.read()
        temp_zero = temp_zero_str.split("Q")
        for i in range(len(temp_zero) - 1):
            load_zero_point.append(int(temp_zero[i]))
    index = 0
    for var_name in list_Name:
        temp_load_numpy = np.load("../SaveModel/ImportAverModel/%s.ndim.npy" % (var_name))
        tensor_load_numpy = torch.tensor(temp_load_numpy)
        test_para = Quan.dequantize_tensor(load_scale[index], tensor_load_numpy, load_zero_point[index])
        index += 1
        model.state_dict()[var_name].copy_(test_para)


# 加载量化存储模型
def quan_No_load_model():
    with open("../SaveModel/ImportQuanSaveModelEdge/scale.txt", 'r') as f:
        temp_scale_str = f.read()
        temp_scale = temp_scale_str.split("Q")
        for i in range(len(temp_scale) - 1):
            trans_scale_np = np.array(temp_scale[i], dtype=float)
            trans_scale_tensor = torch.tensor(trans_scale_np)
            load_scale.append(trans_scale_tensor)
        f.close()
    with open("../SaveModel/ImportQuanSaveModelEdge/zero_point.txt", 'r') as f:
        temp_zero_str = f.read()
        temp_zero = temp_zero_str.split("Q")
        for i in range(len(temp_zero) - 1):
            load_zero_point.append(int(temp_zero[i]))
    index = 0
    for var_name in list_Name:
        temp_load_numpy = np.load("../SaveModel/ImportQuanSaveModelEdge/%s.ndim.npy" % (var_name))
        tensor_load_numpy = torch.tensor(temp_load_numpy)
        test_para = Quan.dequantize_tensor(load_scale[index], tensor_load_numpy, load_zero_point[index])
        index += 1
        model.state_dict()[var_name].copy_(test_para)


# 存储运行时间
def saveTime(time):
    with open("../Savetime/import_quan_runingTime_pruning.txt", 'a') as fwtime:
        fwtime.write(str(float(time)) + '-')
    fwtime.close()

# 存储loss值
def saveLossValue():
    with open("../SaveLossText/import_quan_SaveLossPruning.txt", 'a') as fwriteLoss:
        for lossValue in loss_list:
            fwriteLoss.write(str(lossValue) + "-")
    fwriteLoss.close()

#对比Loss值作为选择性更新依据
def computer_loss():
    temp_loss=loss_list[-1]
    with open("../SaveTransInfor/EdgeOnetrans.txt", 'r') as trans_info_read:
        trans_value_str=trans_info_read.read()
    trans_value=trans_value_str.split("-")
    trans_value=trans_value[:len(trans_value)-1]
    if len(trans_value)==0:
        last_value=100
    else:
        last_value=trans_value[-1]
    print("cur_Loss："+str(temp_loss)+"   last_Loss："+str(last_value))
    if temp_loss<float(last_value):
        with open("../SaveTransInfor/EdgeOnetrans.txt",'a') as loss_write:
            loss_write.write(str(temp_loss)+"-")
        wirte_sign(1)
        return 1
    else:
        wirte_sign(0)
        return 0

def wirte_sign(para):
    with open("../SaveTransInfor/EdgeOne_sign.txt",'a') as swrite:
        if para==1:
            swrite.write("T-")
        else:
            swrite.write("F-")

def read_sign_data():
    with open("./SaveComputerSign/pruning.txt",'r') as read_sign:
        result_str=read_sign.read()
    read_sign.close()
    result_value=result_str.split("-")
    if result_value[-2]=="T":
        return 1
    else:
        return 0

if __name__ == "__main__":
    path = "../SaveModel/ImportAverModel"
    before_train = datetime.datetime.now()
    getName()
    if os.listdir(path):
        if read_sign_data() == 1:
            #print("Aver获取Model")
            quan_load_model()
        else:
            #print("本地获取Model")
            quan_No_load_model()
    train(10)
    if computer_loss() == 1:
        print("EdgeOne 传输 训练权重参量")
    else:
        print("EdgeOne 未传输 训练权重参量")
    quan_save_model()
    saveLossValue()
    after_train = datetime.datetime.now()
    saveTime((after_train - before_train).total_seconds())
