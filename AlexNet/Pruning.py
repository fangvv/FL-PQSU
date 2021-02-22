import torch
import torch.nn as nn
import numpy as np
from AlexNet import AlexNet
import datetime
import Quantification as Quan

prunig_lay=[64,128,256,256,256,3300,3500]
#构建剪枝前后模型
model=AlexNet()
model_new=AlexNet(prunig_lay)

list_Name=[]
save_scale=[]
save_zero_point=[]
load_scale=[]
load_zero_point=[]

def getName():
    for list in model.state_dict():
        list_Name.append(list)

def pruning():
    lay_id=0
    cfg_mask=[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if out_channels == prunig_lay[lay_id]:
                cfg_mask.append(torch.ones(out_channels))
                lay_id += 1
                continue
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            # 给出重要性的排序，从小到大
            arg_min = np.argsort(L1_norm)
            # print(arg_min)
            arg_max_rev = arg_min[::-1][:prunig_lay[lay_id]]
            # print(arg_max_rev)
            #判断选择的层数是否和给定的相同
            assert arg_max_rev.size == prunig_lay[lay_id], "size of arg_max_rev not correct"
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            lay_id += 1

    start_mask = torch.ones(1)
    layer_id_in_cfg = 0
    for [m0, m1] in zip(model.modules(), model_new.modules()):
        if isinstance(m0, nn.Conv2d):
            end_mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            start_mask=end_mask
            layer_id_in_cfg=layer_id_in_cfg+1
        if isinstance(m0, nn.Linear):
            new_out_channels=m1.weight.data.shape[0]
            new_in_channels = m1.weight.data.shape[1]
            weight_copy_liner = m0.weight.data.clone()
            weight_copy_liner = weight_copy_liner.abs().numpy()
            #全连接层从小到大排序
            weight_copy_liner_sum = np.sum(weight_copy_liner, axis=1)
            liner_min = np.argsort(weight_copy_liner_sum)
            liner_max = liner_min[::-1]
            for index in range(new_out_channels):
                m1.weight.data[index]=m0.weight.data[liner_max[index]][:new_in_channels].clone()

def saveModel():
    for name in list_Name:
        temp_num=model_new.state_dict()[name].numpy()
        np.save("../SaveModel/AverModel/%s.ndim"%(name),temp_num)


#加载未量化模型
def loadModel():
    for var_name in list_Name:
        temp_load_numpy = np.load("../SaveModel/Same_model/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        model.state_dict()[var_name].copy_(tensor_load)


#加载量化模型
def quan_load_model():
    with open("../SaveModel/Same_model/scale.txt", 'r') as f:
        temp_scale_str = f.read()
        temp_scale = temp_scale_str.split("Q")
        for i in range(len(temp_scale) - 1):
            trans_scale_np = np.array(temp_scale[i], dtype=float)
            trans_scale_tensor = torch.tensor(trans_scale_np)
            load_scale.append(trans_scale_tensor)
        f.close()
    with open("../SaveModel/Same_model/zero_point.txt", 'r') as f:
        temp_zero_str = f.read()
        temp_zero = temp_zero_str.split("Q")
        for i in range(len(temp_zero) - 1):
            load_zero_point.append(int(temp_zero[i]))
    index = 0
    for var_name in list_Name:
        temp_load_numpy = np.load("../SaveModel/Same_model/%s.ndim.npy" % (var_name))
        tensor_load_numpy = torch.tensor(temp_load_numpy)
        test_para = Quan.dequantize_tensor(load_scale[index], tensor_load_numpy, load_zero_point[index])
        index += 1
        model.state_dict()[var_name].copy_(test_para)


#量化模型
def quan_save_model():
    for varName in list_Name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(model_new.state_dict()[varName])
        model_new.state_dict()[varName].copy_(temp_tensor)
        temp_list = model_new.state_dict()[varName].type(dtype=torch.uint8).numpy()
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


#存储运行时间
def saveTime(time):
    with open("../Savetime/import_quan_pruning_run_time.txt",'a') as fwtime:
        fwtime.write(str(float(time))+'-')
    fwtime.close()


def creat_org_infor():
    with open("../SaveTransInfor/EdgeOne_70_trans.txt",'w') as one_creat:
        one_creat.write("")
    one_creat.close()

    with open("../SaveTransInfor/EdgeTwo_70_trans.txt",'w') as two_creat:
        two_creat.write("")
    two_creat.close()

    with open("../SaveTransInfor/EdgeThree_70_trans.txt",'w') as three_creat:
        three_creat.write("")
    three_creat.close()
    with open("./SaveComputerSign/pruning_30.txt", 'w') as sign_write:
        sign_write.write("T-")
    sign_write.close()

if __name__=="__main__":
    before_time=datetime.datetime.now()
    getName()
    quan_load_model()
    pruning()
    quan_save_model()
    creat_org_infor()
    after_time=datetime.datetime.now()
    saveTime((after_time-before_time).total_seconds())