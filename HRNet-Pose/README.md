# HRNet

## 目录

- [HRNet](#hrnet)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备环境与数据](#3-准备环境与数据)
    - [3.1 准备开发环境](#31-准备开发环境)
      - [3.1.1 开发主机准备：](#311-开发主机准备)
      - [3.1.2 SDK软件包下载：](#312-sdk软件包下载)
      - [3.1.3 创建docker开发环境：](#313-创建docker开发环境)
    - [3.2 准备模型与数据](#32-准备模型与数据)
      - [3.2.1 准备模型](#321-准备模型)
  - [4. 模型转换](#4-模型转换)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成int8 BModel](#42-生成int8-bmodel)
  - [5. 推理测试](#5-推理测试)
    - [5.1 环境配置](#51-环境配置)
      - [5.1.1 x86 SC5](#511-x86-sc5)
    - [5.2 Python例程推理](#52-python例程推理)
  - [6. SE5平台部署](#6-se5平台部署)
    - [6.1 环境配置](#61-环境配置)
    - [6.2 运行](#62-运行)
  - [7. 参考链接与文献](#7-参考链接与文献)


## 1. 简介

在人体姿态识别这类的任务中，需要生成一个高分辨率的heatmap来进行关键点检测。这就与一般的网络结构比如VGGNet的要求不同，因为VGGNet最终得到的feature map分辨率很低，损失了空间结构。HRNet是将不同分辨率的feature map进行并联，在并联的基础上，添加不同分辨率feature map之间的交互(fusion)。


**参考repo:** [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation)

**参考onnx:** [HRNet](https://github.com/open-mmlab/mmpose)

**例程特性：**  
- 支持FP32 INT8 BModel模型编译及推理；
- 支持INT8量化推理；
- 支持batch_size=1、batch_size=4的模型推理；
- 支持python进行前后处理

**适配版本** 
- 支持SOPHON SDK 2.7.0 及以上版本 
- 支持 SE5 


## 2. 数据集

可以使用如下数据集：
https://cocodataset.org/#home

## 3. 准备环境与数据

### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，建议使用我们提供的基于Ubuntu16.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### 3.1.1 开发主机准备：

- 开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机，运行内存建议12GB以上

- 安装docker：参考《[官方教程](https://docs.docker.com/engine/install/)》，若已经安装请跳过

#### 3.1.2 SDK软件包下载：

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://developer.sophgo.com/site/index/material/11/44.html)，Ubuntu 16.04 with Python 3.7

  ```bash
  wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip
  ```

- SDK软件包：[点击前往官网下载SDK软件包](https://developer.sophgo.com/site/index/material/17/45.html)

  ```bash
  wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/05/31/11/bmnnsdk2_bm1684_v2.7.0_20220531patched.zip
  ```

#### 3.1.3 创建docker开发环境：

- 加载docker镜像:

```bash
docker load -i bmnnsdk2-bm1684-ubuntu.docker
```

- 解压缩SDK：

```bash
tar zxvf bmnnsdk2-bm1684_v2.7.0.tar.gz
```

- 创建docker容器，SDK将被挂载映射到容器内部供使用：

```bash
cd bmnnsdk2-bm1684_v2.7.0
# 若您没有执行前述关于docker命令免root执行的配置操作，需在命令前添加sudo
./docker_run_bmnnsdk.sh
```

- 进入docker容器中安装库：

```bash
# 进入容器中执行
cd  /workspace/scripts/
./install_lib.sh nntc
```

- 设置环境变量：

```bash
# 配置环境变量，这一步会安装一些依赖库，并导出环境变量到当前终端
# 导出的环境变量只对当前终端有效，每次进入容器都需要重新执行一遍，或者可以将这些环境变量写入~/.bashrc，这样每次登录将会自动设置环境变量
source envsetup_pcie.sh
```

### 3.2 准备模型与数据

模型来源于：https://github.com/open-mmlab/mmpose，可以从[nas](http://219.142.246.77:65000/sharing/wCwazqhX4)下载模型

#### 3.2.1 准备模型

模型是训练过的，所以可以拿来直接进行推理。

## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。

### 4.1 生成FP32 BModel

进入本例程的工作目录`./scripts`后，需要先运行`bash ./download_from_nas.sh`,再通过运行`bash ./gen_fp32bmodel.sh`，使用bmneto编译生成FP32 BModel。

上述脚本会在`compilation`下生成转换好的bmodel文件，compilation.bmodel

除此之外，还会生产对应的对比文件，可使用`bm_model.bin --info {path_of_bmodel}`查看的模型具体信息。

另外也可以运行`bash ./download_bmodel.sh` 下载编译好的bmodel

### 4.2 生成int8 BModel

进入本例程的工作目录`./scripts`后，需要先运行`bash ./download_from_nas.sh`,再通过运行`bash ./gen_int8bmodel.sh`，编译生成int8 BModel。

在SOPHON SDK 3.0中，还可以使用fake 数据做量化，参考脚本`gen_int8bmodel_fake.sh`。  


除此之外，还会生产对应的对比文件，可使用`bm_model.bin --info {path_of_bmodel}`查看的模型具体信息。

这样直接生成的int8 bmodel由于使用dummy data作为量化数据集，会导致精度较低，可以通过下列方法进行调优

1. 使用真实数据集生成 lmdb 量化数据集进行量化
2. 加大量化迭代次数
3. 量化时将网络输出前的一小段网络使用fp32进行推理

后续也将上传一个调优过后的int8 bmodel

## 5. 推理测试

### 5.1 环境配置

#### 5.1.1 x86 SC5

对于x86 SC5平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成。

### 5.2 Python例程推理

- 环境配置

由于Python例程用到sail库，需安装Sophon Inference：

```bash
# 确认平台及python版本，然后进入相应目录，比如x86平台，python3.7
cd $REL_TOP/lib/sail/python3/pcie/py37
pip3 install sophon-x.x.x-py3-none-any.whl
```

Python代码无需编译，无论是x86 SC5平台还是arm SE5平台配置好环境之后就可直接运行。

> **使用bm_opencv解码的注意事项：** 默认使用原生opencv，若使用bm_opencv解码可能会导致推理结果的差异。若要使用bm_opencv可添加环境变量如下：

```bash
export PYTHONPATH=$PYTHONPATH:$REL_TOP/lib/opencv/x86/opencv-python/
```

测试命令：
因为有默认参数，所以直接运行即可

```bash
python3 bmodel.py
```
生成的图片位于`vis_results` 目录下


## 6. SE5平台部署  

**注意使用 python3.8 版本的 sophon sail**  

### 6.1 环境配置 

```sh
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/
export PYTHONPATH=$PYTHONPATH:/system/lib
``` 
###  6.2 运行  

```sh
python3 bmodel.py
```


## 7. 参考链接与文献
GitHub： https://github.com/HRNet/HRNet-Human-Pose-Estimation
mmpose：https://github.com/open-mmlab/mmpose
DataSet： https://cocodataset.org/#home
