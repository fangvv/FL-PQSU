# FL-PQSU

This is the source code for our paper: **Accelerating Federated Learning for IoT in Big Data Analytics with Pruning, Quantization and Selective Updating**. A brief introduction of this work is as follows:

> The ever-increasing number of Internet of Things (IoT) devices are continuously generating huge masses of data, but the current cloud-centric approach for IoT big data analysis has raised public concerns on both data privacy and network cost. Federated learning (FL) recently emerges as a promising technique to accommodate these concerns, by means of learning a global model by aggregating local updates from multiple devices without sharing the privacy-sensitive data. However, IoT devices usually have constrained computation resources and poor network connections, making it infeasible or very slow to train deep neural networks (DNNs) by following the FL pattern. To address this problem, we propose a new efficient FL framework called FL-PQSU in this paper. It is composed of 3-stage pipeline: structured Pruning, weight Quantization and Selective Updating, that work together to reduce the costs of computation, storage, and communication to accelerate the FL training process. We study FL-PQSU using popular DNN models (AlexNet, VGG16) and publicly available datasets (MNIST, CIFAR10), and demonstrate that it can well control the training overhead while still guaranteeing the learning performance.

> 物联网（IoT）设备数量的持续增长不断产生海量数据，但当前以云计算为核心的物联网大数据分析方法引发了公众对数据隐私和网络成本的双重担忧。联邦学习（FL）作为一种新兴技术，通过聚合多设备本地更新而非共享隐私敏感数据来构建全局模型，为应对这些问题提供了可行方案。然而，物联网设备通常计算资源受限且网络连接状况不佳，遵循联邦学习模式训练深度神经网络（DNN）往往不可行或效率低下。为解决这一问题，本文提出名为FL-PQSU的新型高效联邦学习框架。该框架采用三阶段流水线结构：结构化剪枝（Pruning）、权重量化（Quantization）和选择性更新（Selective Updating），通过协同降低计算、存储和通信成本来加速联邦学习训练过程。我们使用主流深度神经网络模型（AlexNet、VGG16）和公开数据集（MNIST、CIFAR10）对FL-PQSU进行评估，结果表明该框架在保证学习性能的同时，能有效控制训练开销。

This work will be published by IEEE Access, and the paper can be obtained from [here](https://doi.org/10.1109/ACCESS.2021.3063291).

## Required software

- PyTorch
- NumPy
- Pandas
- Torchvision

## Project Structure

```
FL-PQSU/
├── AlexNet/                              # AlexNet model on MNIST
│   ├── AlexNet.py                        # AlexNet model definition (5 conv + 3 fc layers)
│   ├── pre_train.py                      # Pre-train the original model on server
│   ├── Pruning.py                        # Stage 1: structured pruning based on L1-norm
│   ├── Quantification.py                 # Stage 2: 8-bit weight quantization
│   ├── Edge.py                           # Edge device: local training + selective updating
│   └── Server.py                         # Server: model aggregation and accuracy testing
├── VGG/                                  # VGG16 model on CIFAR10
│   ├── VGG.py                            # VGG model definition (with BatchNorm)
│   ├── pre_data_train.py                 # Pre-train the original model on server
│   ├── Pruning.py                        # Stage 1: structured pruning based on L1-norm
│   ├── Quantification.py                 # Stage 2: 8-bit weight quantization
│   ├── Edge.py                           # Edge device: local training + selective updating
│   └── Server.py                         # Server: model aggregation and accuracy testing
├── data.rar                              # Datasets (MNIST and CIFAR10)
└── README.md
```

## Core Modules

### Model Definition (`AlexNet.py` / `VGG.py`)

Defines the DNN models used in the FL framework. The layer configuration can be customized via the `layes`/`cfg` parameter to support pruned models.

**AlexNet** (`AlexNet.py`): 5 convolutional layers + 3 fully-connected layers, default channel config `[96, 256, 384, 384, 256, 4096, 4096]`, input size `1×28×28` (MNIST).

**VGG16** (`VGG.py`): Standard VGG with BatchNorm, default cfg `[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]`, fully-connected layers `[4096, 4096]`, input size `3×32×32` (CIFAR10).

### Pre-training (`pre_train.py` / `pre_data_train.py`)

Pre-trains the original (unpruned) model on the server-side dataset and saves the initial weights. This serves as the starting point for the pruning stage.

| Parameter | AlexNet (MNIST) | VGG16 (CIFAR10) |
|-----------|-----------------|------------------|
| Optimizer | SGD (lr=0.001, momentum=0.9) | SGD (lr=0.1, momentum=0.9) |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |
| Epochs | 10 | 6 |
| Batch size | 30 | 200 |

### Stage 1: Pruning (`Pruning.py`)

Performs structured pruning on convolutional and fully-connected layers based on L1-norm importance. For each layer, filters/neurons with the smallest L1-norm are removed, and a mask is applied to construct a smaller model (`model_new`/`net_new`). The pruned model is then quantized and saved.

**Pruned configurations:**

| Model | Pruned config |
|-------|---------------|
| AlexNet (Edge) | `[64, 128, 256, 256, 256, 3300, 3500]` |
| AlexNet (Server) | `[32, 64, 64, 128, 256, 1900, 2000]` |
| VGG16 | `[16, 16, 'M', 32, 32, 'M', 32, 32, 64, 'M', 64, 64, 64, 'M', 128, 128, 128]`, fc=`[1600, 1600]` |

### Stage 2: Quantization (`Quantification.py`)

Implements 8-bit symmetric quantization to reduce storage and communication costs. Key functions:

| Function | Description |
|----------|-------------|
| `calcScaleZeroPoint(min_val, max_val, num_bits=8)` | Compute the scale and zero-point from the tensor's min/max range. |
| `quantize_tensor(x, num_bits=8)` | Quantize a float tensor to int8, returning `(q_x, scale, zero_point)`. |
| `dequantize_tensor(scale, x, zero_point)` | Dequantize an int8 tensor back to float using `scale * (x - zero_point)`. |

The quantized weights and their scale/zero-point are saved as `.npy` and `.txt` files respectively, reducing the model size by ~4x compared to float32.

### Stage 3: Selective Updating (`Edge.py`)

The edge device performs local training and decides whether to upload its update based on loss comparison. Key workflow:

1. Load the quantized global model (from server or local cache based on the sign flag).
2. Local training for several epochs.
3. Compare the latest loss with the previous round's loss:
   - If `current_loss < previous_loss`: mark as transmittable (`T`), quantize and save the model for uploading.
   - Otherwise: mark as non-transmittable (`F`), skip the upload to save communication.

| Parameter | AlexNet (MNIST) | VGG16 (CIFAR10) |
|-----------|-----------------|------------------|
| Training data per round | 10 batches × 30 samples | 50 batches × 200 samples |
| Epochs per round | 10 | 6 |

### Server Aggregation (`Server.py`)

The server aggregates the quantized models from 3 edge devices by averaging their dequantized weights, tests the aggregated model's accuracy, and then re-quantizes the averaged model for the next round.

- **Aggregation**: `averaged_weight = (weight_edge1 + weight_edge2 + weight_edge3) / 3`
- **Testing**: AlexNet tests on 100 batches of 30 samples; VGG16 tests on 100 batches of 100 samples.
- **Output**: accuracy, running time, and the re-quantized averaged model.

## Usage

```bash
# Install dependencies
pip install torch numpy pandas torchvision

# 1. Pre-train the original model on server
cd AlexNet   # or cd VGG
python pre_train.py   # or python pre_data_train.py

# 2. Pruning stage (structured pruning + quantization)
python Pruning.py

# 3. FL training loop (repeat for multiple rounds):
#    a. Each edge device runs local training with selective updating
python Edge.py

#    b. Server aggregates models from edge devices and tests accuracy
python Server.py
```

**Note:** Before running, extract `data.rar` and place the datasets (MNIST CSV files / CIFAR10) into the corresponding data directories. The code also requires creating the output directories (e.g., `SaveModel/`, `Savetime/`, `SaveLossText/`, `SaveTransInfor/`, etc.) as referenced in the scripts.

## Citation

If you find FL-PQSU useful or relevant to your project and research, please kindly cite our paper:

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

## Stargazers over time

[![Stargazers over time](https://starchart.cc/fangvv/FL-PQSU.svg)](https://starchart.cc/fangvv/FL-PQSU)

## For more

We have another work on [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) for computation offloading optimization in UAV-assisted MEC, and you can simply use [PyTorch](https://pytorch.org/) for implementing deep learning algorithms now.

## Contact

Wenyuan Xu (19120419@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
