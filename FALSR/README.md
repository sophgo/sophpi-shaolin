# FALSR

## 目录

- [FALSR](#falsr)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 测试数据](#2-测试数据)
  - [3. 准备环境与数据](#3-准备环境与数据)
    - [3.1 准备开发环境](#31-准备开发环境)
      - [3.1.1 开发主机准备：](#311-开发主机准备)
      - [3.1.2 SDK软件包下载：](#312-sdk软件包下载)
      - [3.1.3 创建docker开发环境：](#313-创建docker开发环境)
    - [3.2 准备模型](#32-准备模型)
    - [3.3 准备量化集](#33-准备量化集)
  - [4. 模型转换](#4-模型转换)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成INT8 BModel](#42-生成int8-bmodel)
    - [4.2.0 生成LMDB](#420-生成lmdb)
      - [4.2.1 生成FP32 UModel](#421-生成fp32-umodel)
      - [4.2.2 生成INT8 UModel](#422-生成int8-umodel)
      - [4.2.3 生成INT8 BModel](#423-生成int8-bmodel)
  - [5. 部署测试](#5-部署测试)
    - [5.1 环境配置](#51-环境配置)
      - [5.1.1 x86 SC5](#511-x86-sc5)
      - [5.1.2 arm SE5](#512-arm-se5)
    - [5.2 C++例程部署测试](#52-c例程部署测试)
      - [5.2.1 x86平台SC5](#521-x86平台sc5)
      - [5.2.2 arm平台SE5](#522-arm平台se5)
    - [5.3 Python例程部署测试](#53-python例程部署测试)

## 1. 简介

FALSR是小米AI Lab提出的快速、准确且轻量级的图像超分辨率模型。

**参考repo:** [FALSR](https://github.com/xiaomi-automl/FALSR)

**例程特性：**  
- 支持FP32 INT8 BModel模型编译及推理；
- 支持INT8量化推理；
- 支持batch_size=1、batch_size=4的模型推理；
- 支持`python`  `cpp` 进行前后处理

**适配版本** 
- 支持SOPHON SDK 2.7.0 及以上版本 
- 支持 SE5 


## 2. 测试数据

网盘链接: https://pan.baidu.com/s/1UprVQHwqpgrDYfvwojmcwA?pwd=q94r 提取码: q94r 
将链接中的两个视频下载，并保存至`data/videos` 目录下

## 3. 准备环境与数据

### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu16.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### 3.1.1 开发主机准备：

 开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机，运行内存建议12GB以上

- 安装docker：参考《[官方教程](https://docs.docker.com/engine/install/)》，若已经安装请跳过

  ```bash
  # 安装docker
  sudo apt-get install docker.io
  # docker命令免root权限执行
  # 创建docker用户组，若已有docker组会报错，没关系可忽略
  sudo groupadd docker
  # 将当前用户加入docker组
  sudo gpasswd -a ${USER} docker
  # 重启docker服务
  sudo service docker restart
  # 切换当前会话到新group或重新登录重启X会话
  newgrp docker
  ```

#### 3.1.2 SDK软件包下载：

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://sophon.cn/drive/44.html)，Ubuntu 16.04 with Python 3.7

  ```bash
  wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip
  ```

- SDK软件包：[点击前往官网下载SDK软件包](https://sophon.cn/drive/45.html)，BMNNSDK 2.7.0 patched

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
  source envsetup_cmodel.sh
  ```

### 3.2 准备模型

从[FALSR pretrained_model](https://github.com/xiaomi-automl/FALSR/tree/master/pretrained_model)中下载所需要的.pb模型，并将其放入`data/models`目录下。
链接中包含了FALSR-A.pb，FALSR-B.pb，FALSR-C.pb三种不同复杂度的模型，我们选择FALSR-A.pb作为演示的模型。

### 3.3 准备量化集

不量化模型可跳过本节。

这里以Urban100数据集为例，下载[FALSR dataset](https://github.com/xiaomi-automl/FALSR/tree/master/dataset)中的Urban100，并将其放入`data/dataset`目录下。

## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。下面我们以FALSR-A.pb的情况为例，介绍如何完成模型的转换。

### 4.1 生成FP32 BModel

执行以下命令，使用bmnett编译生成FP32 BModel，如果使用了其他的测试视频请注意修改`gen_fp32_bmodel.sh`中的输入参数shapes与视频长宽保持一致，shapes中第二个shape的长宽是第一个的两倍。
```bash
cd scripts
sh ./gen_fp32_bmodel.sh
```
执行完毕后在`/data/fp32model`目录下会生成`falsr_a_fp32.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info falsr_a_fp32.bmodel`查看的模型具体信息如下:

```bash
bmodel version: B.2.2
chip: BM1684
create time: Thu Jun 16 20:54:33 2022

==========================================
net 0: [falsr_a_fp32]  static
------------
stage 0:
input: input_image_evaluate_y, [1, 270, 480, 1], float32, scale: 1
input: input_image_evaluate_pbpr, [1, 540, 960, 2], float32, scale: 1
output: test_sr_evaluator_i1_b0_g/target, [1, 540, 960, 3], float32, scale: 1
```

### 4.2 生成INT8 BModel

不量化模型可跳过本节。

INT8 BModel的生成需要经历中间格式UModel，即：原始模型→FP32 UModel→INT8 UModel→INT8 BModel。

执行以下命令，将依次调用以下步骤中的脚本，生成INT8 BModel：

```shell
sh ./gen_int8bmodel.sh
```

### 4.2.0 生成LMDB

需要将原始量化数据集转换成lmdb格式，供后续校准量化工具Quantization-tools 使用。更详细信息请参考：[准备LMDB数据集](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/module/chapter4.html#lmdb)。

需要从数据集图片生成LMDB文件，具体操作参见`tools/create_lmdb.py`, 相关操作已被封装在 `scripts/create_lmdb.sh`中，执行如下命令即可：

```
sh ./create_lmdb.sh
```

上述脚本会在`data/lmdb`目录中生成lmdb的文件夹，其中存放着量化好的LMDB文件：`data_1.lmdb`和`data_2.lmdb`。若使用了其他的测试视频，请注意根据模型输入要求修改脚本中`create_lmdb.sh`命令中的`resize_width`和`resize_height`等参数。

#### 4.2.1 生成FP32 UModel

执行以下命令，使用`ufw.tools.tf_to_umodel`生成FP32 UModel，若不指定-D参数，可以在生成prototxt文件以后修改：

```bash
sh ./gen_fp32umodel.sh
```

上述脚本会在`int8model/`下生成`*_bmnetp_test_fp32.prototxt`、`*_bmnetp.fp32umodel`文件，即转换好的FP32 UModel。

#### 4.2.2 生成INT8 UModel

执行以下命令，使用修改后的FP32 UModel文件量化生成INT8 UModel：

```
sh ./gen_int8umodel.sh
```

上述脚本会在`int8model/`下生成`*_bmnett_deploy_fp32_unique_top.prototxt`、`*_bmnett_deploy_int8_unique_top.prototxt`和`*_bmnett.int8umodel`文件，即转换好的INT8 UModel。
该操作中对FP32的Model进行量化，将中间层和输出层精度量化为INT8，注意在该模型中YPbPr的输入层对精度较为敏感，对Y和PbPr两个输入层不进行量化。

#### 4.2.3 生成INT8 BModel

执行以下命令，使用生成的INT8 UModel文件生成INT8 BModel：

```
sh ./int8u2bmodel.sh
```

上述脚本会在`int8model/`下生成`*_int8.bmodel`，即转换好的INT8 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Mon Jun 20 09:49:20 2022

==========================================
net 0: [umodel_net]  static
------------
stage 0:
input: input_image_evaluate_y, [1, 270, 480, 1], float32, scale: 1
input: input_image_evaluate_pbpr, [1, 540, 960, 2], float32, scale: 1
output: test_sr_evaluator_i1_b0_g/target, [1, 540, 960, 3], int8, scale: 0.00787209

```

由于量化模型通常存在精度损失，当使用默认脚本生成的量化模型精度不能满足需求时，可能需要修改量化策略并借助自动量化工具auto-calib寻找最优结果，甚至在必要时需要将某些量化精度损失较大的层单独设置为使用fp32推理，相关调试方法请参考[《量化工具用户开发手册》](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/index.html)。

## 5. 部署测试

下载测试数据后，将测试视频放至`data/videos`，转换好的bmodel文件放置于`data/models`。

> 已经转换好的bmodel文件可从[这里](链接：https://pan.baidu.com/s/1EE_KBYm_8xpFrpZPlyEKMg?pwd=xtqp)下载，提取码：xtqp
>

### 5.1 环境配置

#### 5.1.1 x86 SC5

对于x86 SC5平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成
在运行Python Demo时需要安装SAIL模块，详细步骤请参考[《SAIL用户开发手册》](https://doc.sophgo.com/docs/2.7.0/docs_latest_release/sophon-inference/html/get_start/courses/install.html)的2.3.1小节。

#### 5.1.2 arm SE5

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

### 5.2 C++例程部署测试

#### 5.2.1 x86平台SC5

- 编译

```bash
$ cd cpp
$ make -f Makefile.pcie # 生成falsr_test.pcie
```

- 测试

```bash
 $ ./falsr_test.pcie --bmodel=xxx.bmodel --video_path=path/to/video --result_path=path/to/out_video_dir --tpuid=0 --out_imgs_path=path/to/out_images_dir # use your own falsr bmodel
```

#### 5.2.2 arm平台SE5

对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译

```bash
$ cd cpp
$ make -f Makefile.arm # 生成falsr_test.arm
```

- 将生成的可执行文件及所需的模型和测试视频文件拷贝到盒子中测试

```bash
 $ ./falsr_test.arm --bmodel=xxx.bmodel --video_path=path/to/video --result_path=path/to/out_video_dir --tpuid=0 --out_imgs_path=path/to/out_images_dir # use your own falsr bmodel
```

### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

```bash
$ cd python
$ python3 falsr_scipy.py --bmodel path/to/your_model --video_path path/to/your_video --out_path path/to/out_video 
```

> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。