import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from AlexNet import AlexNet
import pandas as pd

train_data=pd.DataFrame(pd.read_csv("./Mnist_data\server.csv",encoding='utf-8'))
test_data=pd.DataFrame(pd.read_csv("./Mnist_data\mnist_test.csv"))
model=AlexNet()
loss_fc=nn.CrossEntropyLoss()
optimizer=optim.SGD(params=model.parameters(),lr=0.001,momentum=0.9)
list_Name = []

def getName():
    for list in model.state_dict():
        list_Name.append(list)

def saveModel():
    for name in list_Name:
        temp_num=model.state_dict()[name].numpy()
        np.save("../SaveModel/Same_model/%s.ndim"%(name),temp_num)

#训练代码
def train(epoch):
    for j in range(epoch):
        for i in range(10):
            batch_data=train_data.sample(n=30,replace=False)
            batch_y=torch.from_numpy(batch_data.iloc[:,0].values).long()
            batch_x=torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)
            prediction=model.forward(batch_x)
            loss=loss_fc(prediction,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("第%d次epoch第%d训练，loss为%0.3f"%(j,i,loss))

if __name__=="__main__":
    getName()
    train(10)
    saveModel()