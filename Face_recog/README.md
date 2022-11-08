# Face_recog

## 目录

[TOC]

## 1. 简介

本章采用retinaFace模型进行人脸检测，并采用resnet模型进行人脸识别，并已提供了转换后的bmodel模型，可以直接进行模型部署测试。



## 2. 部署测试

### 2.1 环境配置

#### 2.1.1 x86 SC5

对于x86 SC5平台，程序执行所需的环境变量请参考官方网站入门指南[technical center (sophgo.com)](https://developer.sophgo.com/site/index/document/all/all.html)，选择当前版本SDK对应指南进行配置。注意：在运行Python Demo时需要安装SAIL模块，详细步骤请参考[《SAIL用户开发手册》](https://doc.sophgo.com/docs/2.7.0/docs_latest_release/sophon-inference/html/get_start/courses/install.html)的2.3.1小节。

#### 2.1.2 arm SE5

对于arm SE5平台，内部已经集成了相应的SDK运行库包，位于/system目录下，只需设置环境变量即可。

```bash
# 设置环境变量
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
export PYTHONPATH=$PYTHONPATH:/system/lib
```

您可能需要安装numpy包，以在Python中使用OpenCV和SAIL：

```bash
# 请指定numpy版本为1.17.2
sudo pip3 install numpy==1.17.2
```

在运行Python Demo时需要安装SAIL模块，详细步骤请参考[《SAIL用户开发手册》](https://doc.sophgo.com/docs/2.7.0/docs_latest_release/sophon-inference/html/get_start/courses/install.html)的2.3.1小节。

### 2.2 python 例程部署测试

```bash
# 下载所需数据集
cd scripts
bash download_datasets.sh

# 运行程序
cd ../python
python3 Face_recogion.py
```

