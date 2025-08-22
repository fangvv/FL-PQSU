## FL-PQSU

This is the source code for our paper: **Accelerating Federated Learning for IoT in Big Data Analytics with Pruning, Quantization and Selective Updating**. A brief introduction of this work is as follows:

> The ever-increasing number of Internet of Things (IoT) devices are continuously generating huge masses of data, but the current cloud-centric approach for IoT big data analysis has raised public
concerns on both data privacy and network cost. Federated learning (FL) recently emerges as a promising technique to accommodate these concerns, by means of learning a global model by aggregating local
updates from multiple devices without sharing the privacy-sensitive data. However, IoT devices usually have constrained computation resources and poor network connections, making it infeasible or very slow
to train deep neural networks (DNNs) by following the FL pattern. To address this problem, we propose a new efficient FL framework called FL-PQSU in this paper. It is composed of 3-stage pipeline: structured
Pruning, weight Quantization and Selective Updating, that work together to reduce the costs of computation, storage, and communication to accelerate the FL training process. We study FL-PQSU using popular DNN
models (AlexNet, VGG16) and publicly available datasets (MNIST, CIFAR10), and demonstrate that it can well control the training overhead while still guaranteeing the learning performance.

> 物联网（IoT）设备数量的持续增长不断产生海量数据，但当前以云计算为核心的物联网大数据分析方法引发了公众对数据隐私和网络成本的双重担忧。联邦学习（FL）作为一种新兴技术，通过聚合多设备本地更新而非共享隐私敏感数据来构建全局模型，为应对这些问题提供了可行方案。然而，物联网设备通常计算资源受限且网络连接状况不佳，遵循联邦学习模式训练深度神经网络（DNN）往往不可行或效率低下。为解决这一问题，本文提出名为FL-PQSU的新型高效联邦学习框架。该框架采用三阶段流水线结构：结构化剪枝（Pruning）、权重量化（Quantization）和选择性更新（Selective Updating），通过协同降低计算、存储和通信成本来加速联邦学习训练过程。我们使用主流深度神经网络模型（AlexNet、VGG16）和公开数据集（MNIST、CIFAR10）对FL-PQSU进行评估，结果表明该框架在保证学习性能的同时，能有效控制训练开销。

This work will be published by IEEE Access, and the paper can be obtained from [here](https://doi.org/10.1109/ACCESS.2021.3063291).

## Required software

PyTorch

## Citation

      @ARTICLE{9366879,
      author={Xu, Wenyuan and Fang, Weiwei and Ding, Yi and Zou, Meixia and Xiong, Naixue},
      journal={IEEE Access}, 
      title={Accelerating Federated Learning for IoT in Big Data Analytics With Pruning, Quantization and Selective Updating}, 
      year={2021},
      volume={9},
      number={},
      pages={38457-38466},
      doi={10.1109/ACCESS.2021.3063291}
	  }

## Contact

Wenyuan Xu (19120419@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.

