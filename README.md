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

