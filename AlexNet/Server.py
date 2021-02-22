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

prunig_lay=[32,64,64,128,256,1900,2000]
model=AlexNet(prunig_lay)

test_data=pd.DataFrame(pd.read_csv("./Mnist_data/mnist_test.csv"))
list_Name=[]
acc=[]
x=[]

save_scale=[]
save_zero_point=[]

one_zero_point=[]
one_scale=[]
two_zero_point=[]
two_scale=[]
three_zero_point=[]
three_scale=[]

def getName():
    for list in model.state_dict():
        list_Name.append(list)

#加载未量化模型
def load_model():
    for var_name in list_Name:
        temp_load_numpy_one = np.load("../SaveModel/SaveModelEdgeOne/%s.ndim.npy" % (var_name))
        temp_load_numpy_two = np.load("../SaveModel/SaveModelEdgeTwo/%s.ndim.npy" % (var_name))
        temp_load_numpy_three = np.load("../SaveModel/SaveModelEdgeThree/%s.ndim.npy" % (var_name))
        temp_load_numpy=(temp_load_numpy_three+temp_load_numpy_two+temp_load_numpy_one)/3
        tensor_load=torch.tensor(temp_load_numpy)
        model.state_dict()[var_name].copy_(tensor_load)

#测试合并模型精度
def test():
    sum=0
    with torch.no_grad():
        for i in range(100):
            batch_data = test_data.sample(n=30, replace=False)
            batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
            batch_y = batch_data.iloc[:, 0].values
            prediction = np.argmax(model(batch_x).numpy(), axis=1)
            accurcy = np.mean(prediction == batch_y)
            sum+=accurcy
            acc.append(accurcy*100)
            x.append(i)
            print("第%d组测试集，准确率为%.3f" % (i, accurcy))
    acc_aver=(sum/100)*100
    saveAcc(acc_aver)
    print("精度为%f："%(acc_aver))
    return acc_aver

#保存模型
def saveModel():
    for name in list_Name:
        temp_num=model.state_dict()[name].numpy()
        np.save("../SaveModel/AverModel/%s.ndim"%(name),temp_num)

# 量化保存模型
def quan_save_model():
    for varName in list_Name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(model.state_dict()[varName])
        model.state_dict()[varName].copy_(temp_tensor)
        temp_list = model.state_dict()[varName].type(dtype=torch.uint8).numpy()
        np.save("../SaveModel/ImportAverModel/%s.ndim.npy" % (varName), temp_list)
        save_scale.append(temp_scale.item())
        save_zero_point.append(temp_zeropoint)

    with open("../SaveModel/ImportAverModel/scale.txt", 'w') as f:
        for temp_np in save_scale:
            f.write(str(temp_np) + "Q")
        f.close()

    with open("../SaveModel/ImportAverModel/zero_point.txt", 'w') as f:
        for temp_zero in save_zero_point:
            f.write(str(temp_zero) + "Q")
        f.close()


# 加载量化模型
def quan_load_model():
    #加载Edge_one的scale和zero_point
    with open("../SaveModel/ImportQuanSaveModelEdgeOne/scale.txt", 'r') as f:
        one_temp_scale_str = f.read()
        one_temp_scale = one_temp_scale_str.split("Q")
        for i in range(len(one_temp_scale) - 1):
            one_trans_scale_np = np.array(one_temp_scale[i], dtype=float)
            one_trans_scale_tensor = torch.tensor(one_trans_scale_np)
            one_scale.append(one_trans_scale_tensor)
        f.close()
    with open("../SaveModel/ImportQuanSaveModelEdgeOne/zero_point.txt", 'r') as f:
        one_temp_zero_str = f.read()
        one_temp_zero = one_temp_zero_str.split("Q")
        for i in range(len(one_temp_zero) - 1):
            one_zero_point.append(int(one_temp_zero[i]))

    # 加载Edge_two的scale和zero_point
    with open("../SaveModel/ImportQuanSaveModelEdgeTwo/scale.txt", 'r') as f:
        two_temp_scale_str = f.read()
        two_temp_scale = two_temp_scale_str.split("Q")
        for i in range(len(two_temp_scale) - 1):
            two_trans_scale_np = np.array(two_temp_scale[i], dtype=float)
            two_trans_scale_tensor = torch.tensor(two_trans_scale_np)
            two_scale.append(two_trans_scale_tensor)
        f.close()
    with open("../SaveModel/ImportQuanSaveModelEdgeTwo/zero_point.txt", 'r') as f:
        two_temp_zero_str = f.read()
        two_temp_zero = two_temp_zero_str.split("Q")
        for i in range(len(two_temp_zero) - 1):
            two_zero_point.append(int(two_temp_zero[i]))

    # 加载Edge_three的scale和zero_point
    with open("../SaveModel/ImportQuanSaveModelEdgeThree/scale.txt", 'r') as f:
        three_temp_scale_str = f.read()
        three_temp_scale = three_temp_scale_str.split("Q")
        for i in range(len(three_temp_scale) - 1):
            three_trans_scale_np = np.array(three_temp_scale[i], dtype=float)
            three_trans_scale_tensor = torch.tensor(three_trans_scale_np)
            three_scale.append(three_trans_scale_tensor)
        f.close()
    with open("../SaveModel/ImportQuanSaveModelEdgeThree/zero_point.txt", 'r') as f:
        three_temp_zero_str = f.read()
        three_temp_zero = three_temp_zero_str.split("Q")
        for i in range(len(three_temp_zero) - 1):
            three_zero_point.append(int(three_temp_zero[i]))

    index = 0
    for var_name in list_Name:
        temp_load_numpy_one= np.load("../SaveModel/ImportQuanSaveModelEdgeOne/%s.ndim.npy" % (var_name))
        tensor_load_numpy_one = torch.tensor(temp_load_numpy_one)
        test_para_one = Quan.dequantize_tensor(one_scale[index], tensor_load_numpy_one, one_zero_point[index])

        temp_load_numpy_two = np.load("../SaveModel/ImportQuanSaveModelEdgeTwo/%s.ndim.npy" % (var_name))
        tensor_load_numpy_two = torch.tensor(temp_load_numpy_two)
        test_para_two = Quan.dequantize_tensor(two_scale[index], tensor_load_numpy_two, two_zero_point[index])

        temp_load_numpy_three = np.load("../SaveModel/ImportQuanSaveModelEdgeThree/%s.ndim.npy" % (var_name))
        tensor_load_numpy_three = torch.tensor(temp_load_numpy_three)
        test_para_three = Quan.dequantize_tensor(three_scale[index], tensor_load_numpy_three, three_zero_point[index])

        index += 1
        model.state_dict()[var_name].copy_((test_para_one+test_para_two+test_para_three)/3)

# 存储运行时间
def saveTime(time):
    with open("../Savetime/import_quan_runingTime_pruning70_serverAve.txt", 'a') as fwtime:
        fwtime.write(str(float(time)) + '-')
    fwtime.close()

def saveAcc(acc):
    with open("../SaveACC/import_quan_pruning70_Acc.txt", 'a') as fwtime:
        fwtime.write(str(float(acc)) + '-')
    fwtime.close()

if __name__=="__main__":
    before_time=datetime.datetime.now()
    getName()
    quan_load_model()
    test()
    quan_save_model()
    after_time=datetime.datetime.now()
    saveTime((after_time-before_time).total_seconds())