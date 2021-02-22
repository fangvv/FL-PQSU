import sys
sys.path.append("/mnt/FLVGG/VGG")
import torch
import torchvision
import torchvision.transforms as transforms
from VGG import vgg
import numpy as np
import datetime
import Quantification as Quan

prunig_lay=[16, 16, 'M', 32, 32, 'M', 32, 32, 64, 'M', 64, 64, 64, 'M', 128, 128, 128]
net = vgg(depth=16, cfg=prunig_lay,fullLayers=[1600, 1600])
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.cuda()
acc_data = []
list_name=[]
save_scale = []
save_zero_point = []
one_zero_point = []
one_scale = []
two_zero_point = []
two_scale = []
three_zero_point = []
three_scale = []
load_scale=[]
load_zero_point=[]

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./CifarTestData', train=False,
                                       download=True, transform=transform)


def getName():
    for name in net.state_dict():
        list_name.append(name)


def load_model():
    for var_name in list_name:
        temp_load_numpy_one = np.load("./SaveModel/SaveEdgeOneModel/%s.ndim.npy" % (var_name))
        temp_load_numpy_two = np.load("./SaveModel/SaveEdgeTwoModel/%s.ndim.npy" % (var_name))
        temp_load_numpy_three = np.load("./SaveModel/SaveEdgeThreeModel/%s.ndim.npy" % (var_name))
        temp_load_numpy=(temp_load_numpy_three+temp_load_numpy_two+temp_load_numpy_one)/3
        tensor_load=torch.tensor(temp_load_numpy)
        net.state_dict()[var_name].copy_(tensor_load)

        
def save_model():
    for name in list_name:
        temp_num=net.state_dict()[name].cpu().numpy()
        np.save("./SaveModel/AverModel/%s.ndim"%(name),temp_num)


# 量化存储模型
def quan_save_model():
    for varName in list_name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(net.state_dict()[varName])
        net.state_dict()[varName].copy_(temp_tensor)
        temp_list = net.state_dict()[varName].type(dtype=torch.uint8).cpu().numpy()
        np.save("./SaveModel/Quan_AverModel/%s.ndim.npy" % (varName), temp_list)
        save_scale.append(temp_scale.item())
        save_zero_point.append(temp_zeropoint)

    with open("./SaveModel/Quan_AverModel/scale.txt", 'w') as f:
        for temp_np in save_scale:
            f.write(str(temp_np) + "Q")
        f.close()

    with open("./SaveModel/Quan_AverModel/zero_point.txt", 'w') as f:
        for temp_zero in save_zero_point:
            f.write(str(temp_zero) + "Q")
        f.close()


#加载量化模型
def quan_load_model():
    # 加载Edge_one的scale和zero_point
    with open("./SaveModel/Quan_SaveEdgeOneModel/scale.txt", 'r') as f:
        one_temp_scale_str = f.read()
        one_temp_scale = one_temp_scale_str.split("Q")
        for i in range(len(one_temp_scale) - 1):
            one_trans_scale_np = np.array(one_temp_scale[i], dtype=float)
            one_trans_scale_tensor = torch.tensor(one_trans_scale_np)
            one_scale.append(one_trans_scale_tensor)
        f.close()

    with open("./SaveModel/Quan_SaveEdgeOneModel/zero_point.txt", 'r') as f:
        one_temp_zero_str = f.read()
        one_temp_zero = one_temp_zero_str.split("Q")
        for i in range(len(one_temp_zero) - 1):
            one_zero_point.append(int(one_temp_zero[i]))
        f.close()

    # 加载Edge_two的scale和zero_point
    with open("./SaveModel/Quan_SaveEdgeTwoModel/scale.txt", 'r') as f:
        two_temp_scale_str = f.read()
        two_temp_scale = two_temp_scale_str.split("Q")
        for i in range(len(two_temp_scale) - 1):
            two_trans_scale_np = np.array(two_temp_scale[i], dtype=float)
            two_trans_scale_tensor = torch.tensor(two_trans_scale_np)
            two_scale.append(two_trans_scale_tensor)
        f.close()

    with open("./SaveModel/Quan_SaveEdgeTwoModel/zero_point.txt", 'r') as f:
        two_temp_zero_str = f.read()
        two_temp_zero = two_temp_zero_str.split("Q")
        for i in range(len(two_temp_zero) - 1):
            two_zero_point.append(int(two_temp_zero[i]))
        f.close()

    # 加载Edge_three的scale和zero_point
    with open("./SaveModel/Quan_SaveEdgeThreeModel/scale.txt", 'r') as f:
        three_temp_scale_str = f.read()
        three_temp_scale = three_temp_scale_str.split("Q")
        for i in range(len(three_temp_scale) - 1):
            three_trans_scale_np = np.array(three_temp_scale[i], dtype=float)
            three_trans_scale_tensor = torch.tensor(three_trans_scale_np)
            three_scale.append(three_trans_scale_tensor)
        f.close()

    with open("./SaveModel/Quan_SaveEdgeThreeModel/zero_point.txt", 'r') as f:
        three_temp_zero_str = f.read()
        three_temp_zero = three_temp_zero_str.split("Q")
        for i in range(len(three_temp_zero) - 1):
            three_zero_point.append(int(three_temp_zero[i]))
        f.close()

    index = 0
    for var_name in list_name:
        temp_load_numpy_one = np.load("./SaveModel/Quan_SaveEdgeOneModel/%s.ndim.npy" % (var_name))
        tensor_load_numpy_one = torch.tensor(temp_load_numpy_one)
        test_para_one = Quan.dequantize_tensor(one_scale[index], tensor_load_numpy_one, one_zero_point[index])

        temp_load_numpy_two = np.load("./SaveModel/Quan_SaveEdgeTwoModel/%s.ndim.npy" % (var_name))
        tensor_load_numpy_two = torch.tensor(temp_load_numpy_two)
        test_para_two = Quan.dequantize_tensor(two_scale[index], tensor_load_numpy_two, two_zero_point[index])

        temp_load_numpy_three = np.load("./SaveModel/Quan_SaveEdgeThreeModel/%s.ndim.npy" % (var_name))
        tensor_load_numpy_three = torch.tensor(temp_load_numpy_three)
        test_para_three = Quan.dequantize_tensor(three_scale[index], tensor_load_numpy_three, three_zero_point[index])
        
        net.state_dict()[var_name].copy_((test_para_one+test_para_two+test_para_three)/3)
        index+=1

#获取测试数据
def get_test_data(index):
    test_x = []
    test_y = []
    for i in range(index, index + 100):
        test_x.append(testset.__getitem__(i)[0].numpy())
        test_y.append(testset.__getitem__(i)[1])
    test_x = torch.tensor(test_x).view(-1, 3, 32, 32)
    test_y = torch.tensor(test_y)
    return test_x, test_y


def test():
    with torch.no_grad():
        acc_mean=0
        for w in range(len(test_number)):
            test_x, test_y = get_test_data(w)
            test_x,test_y=test_x.to(device),test_y.to(device)
            prediction = np.argmax(net(test_x).cpu().numpy(), axis=1)
            accurcy = np.mean(prediction == test_y.cpu().numpy())
            acc_mean+=accurcy
            print("第"+str(w)+"次精度为："+str(accurcy))
        print("平均精度为："+str((acc_mean/100)*100))
        saveAcc((acc_mean/100)*100)


def saveAcc(acc):
    with open("./SaveAcc/saveAcc_pruning_quan_imp_0.txt", 'a') as wacc:
        wacc.write(str(acc) + "-")
    wacc.close()


def saveTime(time):
    with open("./SaveTime/saveTime_Server_pruning_quan_imp_0.txt", 'a') as fwtime:
        fwtime.write(str(float(time)) + '-')
    fwtime.close()


if __name__=="__main__":
    before_train = datetime.datetime.now()
    test_number = list(range(0, 10000, 100))
    print(len(test_number))
    getName()
    quan_load_model()
    test()
    quan_save_model()
    after_train = datetime.datetime.now()
    saveTime((after_train - before_train).total_seconds())