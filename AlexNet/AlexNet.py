import torch
import math
import torch.nn as nn

def_lay=[96,256,384,384,256,4096,4096]

class AlexNet(nn.Module):
    def __init__(self,layes=None,init_weight=True):
        super(AlexNet,self).__init__()
        if layes==None:
            layes=def_lay

        self.lays=layes

        #初始化权重
        if init_weight:
            self.init_weights()

        #第1卷积层
        self.conv1=nn.Conv2d(1,layes[0],kernel_size=3,padding=1,stride=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu1=nn.ReLU()

        #第2卷积层
        self.conv2=nn.Conv2d(layes[0],layes[1],kernel_size=3,stride=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu2=nn.ReLU()

        #第3,4,5卷积层
        self.conv3=nn.Conv2d(layes[1],layes[2],kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(layes[2],layes[3],kernel_size=3,stride=1,padding=1)
        self.conv5=nn.Conv2d(layes[3],layes[4],kernel_size=3,stride=1,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu3=nn.ReLU()

        self.fc6=nn.Linear(layes[4]*3*3,layes[5])
        self.fc7=nn.Linear(layes[5],layes[6])
        self.fc8=nn.Linear(layes[6],10)



    def forward(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.pool3(x)
        x=self.relu3(x)
        x=x.view(-1,self.lays[4]*3*3)
        x=self.fc6(x)
        x=self.fc7(x)
        x=self.fc8(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__=="__main__":
    alex=AlexNet()
    print(alex)
