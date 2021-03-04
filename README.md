## FL-PQSU

This is the source code for our paper: **Accelerating Federated Learning for IoT in Big Data Analytics with Pruning, Quantization and Selective Updating**. A brief introduction of this work is as follows:

> The ever-increasing number of Internet of Things (IoT) devices are continuously generating huge masses of data, but the current cloud-centric approach for IoT big data analysis has raised public
concerns on both data privacy and network cost. Federated learning (FL) recently emerges as a promising technique to accommodate these concerns, by means of learning a global model by aggregating local
updates from multiple devices without sharing the privacy-sensitive data. However, IoT devices usually have constrained computation resources and poor network connections, making it infeasible or very slow
to train deep neural networks (DNNs) by following the FL pattern. To address this problem, we propose a new efficient FL framework called FL-PQSU in this paper. It is composed of 3-stage pipeline: structured
Pruning, weight Quantization and Selective Updating, that work together to reduce the costs of computation, storage, and communication to accelerate the FL training process. We study FL-PQSU using popular DNN
models (AlexNet, VGG16) and publicly available datasets (MNIST, CIFAR10), and demonstrate that it can well control the training overhead while still guaranteeing the learning performance.

> 通过剪枝、量化和选择性更新来加速物联网设备上基于联邦计算的DNN模型训练

This work will be published by IEEE Access, and the paper can be obtained from [here](https://doi.org/10.1109/ACCESS.2021.3063291).

## Required software

PyTorch

## Contact

Wenyuan Xu (19120419@bjtu.edu.cn)

Weiwei Fang (fangvv@qq.com)

