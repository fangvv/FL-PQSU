import sys
import torch
import torch.nn as nn
from VGG import vgg
import numpy as np
import datetime
import Quantification as Quan

prunig_lay=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
net_old = vgg(depth=16, fullLayers=[4096, 4096])
net_new = vgg(depth=16,fullLayers=[4096,4096],cfg=prunig_lay)
list_name = []
save_scale=[]
save_zero_point=[]


def getName():
    for name in net_old.state_dict():
        list_name.append(name)


def pruning():
    cfg_mask = []
    layer_id = 0
    for m in net_old.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if out_channels == prunig_lay[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                layer_id += 1
                continue
            weight_copy = m.weight.data.cpu().abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:prunig_lay[layer_id]]
            assert arg_max_rev.size == prunig_lay[layer_id], 'size of arg_max_rev not correct'
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1
        elif isinstance(m, nn.MaxPool2d):
            layer_id += 1

    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    for [m0, m1] in zip(net_old.modules(), net_new.modules()):
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
            start_mask = end_mask
            layer_id_in_cfg = layer_id_in_cfg + 1

        if isinstance(m0, nn.Linear):
            new_out_channels=m1.weight.data.shape[0]
            new_in_channels = m1.weight.data.shape[1]
            weight_copy_liner = m0.weight.data.clone()
            weight_copy_liner = weight_copy_liner.abs().numpy()
            weight_copy_liner_sum = np.sum(weight_copy_liner, axis=1)
            liner_min = np.argsort(weight_copy_liner_sum)
            liner_max = liner_min[::-1]
            for index in range(new_out_channels):
                m1.weight.data[index]=m0.weight.data[liner_max[index]][:new_in_channels].clone()

def saveTime(time):
    with open("./SaveTime/saveTime_pruning_quam_imp.txt", 'a') as fwtime:
        fwtime.write(str(float(time)) + '-')
    fwtime.close()

def loadModel():
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveQuanOrgModel/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        net_old.state_dict()[var_name].copy_(tensor_load)


def save_model():
    for name in list_name:
        temp_np = net_new.state_dict()[name].numpy()
        np.save("./SaveModel/AverModel/%s.ndim" % (name), temp_np)


#存储量化模型
def quan_save_model():
    for varName in list_name:
        temp_tensor, temp_scale, temp_zeropoint = Quan.quantize_tensor(net_new.state_dict()[varName])
        net_new.state_dict()[varName].copy_(temp_tensor)
        temp_list = net_new.state_dict()[varName].type(dtype=torch.uint8).numpy()
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

def create_org_loss_value():
    with open("./SaveTransInfo/No_Data/EdgeOne_Loss_0.txt",'w') as EdgeOne_writeValue:
        EdgeOne_writeValue.write("")
    EdgeOne_writeValue.close()
    
    with open("./SaveTransInfo/No_Data/EdgeTwo_Loss_0.txt",'w') as EdgeTwo_writeValue:
        EdgeTwo_writeValue.write("")
    EdgeTwo_writeValue.close()
    
    with open("./SaveTransInfo/No_Data/EdgeThree_Loss_0.txt",'w') as EdgeThree_writeValue:
        EdgeThree_writeValue.write("")
    EdgeThree_writeValue.close()
        
if __name__=="__main__":
    before_train = datetime.datetime.now()
    getName()
    loadModel()
    pruning()
    quan_save_model()
    after_train = datetime.datetime.now()
    create_org_loss_value()
    saveTime((after_train - before_train).total_seconds())