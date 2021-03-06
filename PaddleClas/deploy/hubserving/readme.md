[English](readme_en.md) | 简体中文

# 基于PaddleHub Serving的服务部署

hubserving服务部署配置服务包`clas`下包含3个必选文件，目录如下：
```
deploy/hubserving/clas/
  └─  __init__.py    空文件，必选
  └─  config.json    配置文件，可选，使用配置启动服务时作为参数传入
  └─  module.py      主模块，必选，包含服务的完整逻辑
  └─  params.py      参数文件，必选，包含模型路径、前后处理参数等参数
```

## 快速启动服务
### 1. 准备环境
```shell
# 安装paddlehub,请安装2.0版本
pip3 install paddlehub==2.0.0b1 --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载推理模型
安装服务模块前，需要准备推理模型并放到正确路径，默认模型路径为：
```
分类推理模型结构文件：./inference/cls_infer.pdmodel
分类推理模型权重文件：./inference/cls_infer.pdiparams
```  

**注意**：
* 模型路径可在`./PaddleClas/deploy/hubserving/clas/params.py`中查看和修改。
  ```python
  cfg.model_file = "./inference/cls_infer.pdmodel"
  cfg.params_file = "./inference/cls_infer.pdiparams"
  ```
* 我们也提供了大量基于ImageNet-1k数据集的预训练模型，模型列表及下载地址详见[模型库概览](../../docs/zh_CN/models/models_intro.md)，也可以使用自己训练转换好的模型。

### 3. 安装服务模块
针对Linux环境和Windows环境，安装命令如下。

* 在Linux环境下，安装示例如下：
```shell
# 安装服务模块：  
hub install deploy/hubserving/clas/
```

* 在Windows环境下(文件夹的分隔符为`\`)，安装示例如下：
```shell
# 安装服务模块：  
hub install deploy\hubserving\clas\
```

### 4. 启动服务
#### 方式1. 命令行命令启动（仅支持CPU）
**启动命令：**  
```shell
$ hub serving start --modules Module1==Version1 \
                    --port XXXX \
                    --use_multiprocess \
                    --workers \
```  

**参数：**  

|参数|用途|  
|-|-|  
|--modules/-m| [**必选**] PaddleHub Serving预安装模型，以多个Module==Version键值对的形式列出<br>*`当不指定Version时，默认选择最新版本`*|  
|--port/-p| [**可选**] 服务端口，默认为8866|  
|--use_multiprocess| [**可选**] 是否启用并发方式，默认为单进程方式，推荐多核CPU机器使用此方式<br>*`Windows操作系统只支持单进程方式`*|
|--workers| [**可选**] 在并发方式下指定的并发任务数，默认为`2*cpu_count-1`，其中`cpu_count`为CPU核数|  

如按默认参数启动服务：  ```hub serving start -m clas_system```  

这样就完成了一个服务化API的部署，使用默认端口号8866。

#### 方式2. 配置文件启动（支持CPU、GPU）
**启动命令：**  
```hub serving start -c config.json```  

其中，`config.json`格式如下：
```json
{
    "modules_info": {
        "clas_system": {
            "init_args": {
                "version": "1.0.0",
                "use_gpu": true,
                "enable_mkldnn": false
            },
            "predict_args": {
            }
        }
    },
    "port": 8866,
    "use_multiprocess": false,
    "workers": 2
}
```

- `init_args`中的可配参数与`module.py`中的`_initialize`函数接口一致。其中，
  - 当`use_gpu`为`true`时，表示使用GPU启动服务。
  - 当`enable_mkldnn`为`true`时，表示使用MKL-DNN加速。
- `predict_args`中的可配参数与`module.py`中的`predict`函数接口一致。

**注意:**  
- 使用配置文件启动服务时，其他参数会被忽略。
- 如果使用GPU预测(即，`use_gpu`置为`true`)，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，如：```export CUDA_VISIBLE_DEVICES=0```，否则不用设置。
- **`use_gpu`不可与`use_multiprocess`同时为`true`**。
- **`use_gpu`与`enable_mkldnn`同时为`true`时，将忽略`enable_mkldnn`，而使用GPU**。

如，使用GPU 3号卡启动串联服务：  
```shell
export CUDA_VISIBLE_DEVICES=3
hub serving start -c deploy/hubserving/clas/config.json
```  

## 发送预测请求
配置好服务端，可使用以下命令发送预测请求，获取预测结果:  

```python tools/test_hubserving.py server_url image_path```  

需要给脚本传递2个必须参数：
- **server_url**：服务地址，格式为  
`http://[ip_address]:[port]/predict/[module_name]`  
- **image_path**：测试图像路径，可以是单张图片路径，也可以是图像集合目录路径。
- **top_k**：[**可选**] 返回前 `top_k` 个 `score` ，默认为 `1`。
- **batch_size**：[**可选**] 以`batch_size`大小为单位进行预测，默认为`1`。
- **resize_short**：[**可选**] 将图像等比例缩放到最短边为`resize_short`，默认为`256`。
- **resize**：[**可选**] 将图像resize到`resize * resize`尺寸，默认为`224`。
- **normalize**：[**可选**] 是否对图像进行normalize处理，默认为`True`。

**注意**：如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸。需要指定`--resize_short=384 --resize=384`。


访问示例：  
```python tools/test_hubserving.py --server_url http://127.0.0.1:8866/predict/clas_system --image_file ./deploy/hubserving/ILSVRC2012_val_00006666.JPEG --top_k 5```

### 返回结果格式说明
返回结果为列表（list），包含top-k个分类结果，以及对应的得分，还有此图片预测耗时，具体如下：
```
list: 返回结果
└─ list: 第一张图片结果
   └─ list: 前k个分类结果，依score递减排序
   └─ list: 前k个分类结果对应的score，依score递减排序
   └─ float: 该图分类耗时，单位秒
```

**说明：** 如果需要增加、删除、修改返回字段，可在相应模块的`module.py`文件中进行修改，完整流程参考下一节自定义修改服务模块。

## 自定义修改服务模块
如果需要修改服务逻辑，你一般需要操作以下步骤：  

- 1、 停止服务  
```hub serving stop --port/-p XXXX```  

- 2、 到相应的`module.py`和`params.py`等文件中根据实际需求修改代码。  
  例如，例如需要替换部署服务所用模型，则需要到`params.py`中修改模型路径参数`cfg.model_file`和`cfg.params_file`。

  修改并安装（`hub install deploy/hubserving/clas/`）完成后，在进行部署前，可通过`python deploy/hubserving/clas/test.py`测试已安装服务模块。

- 3、 卸载旧服务包  
```hub uninstall clas_system```  

- 4、 安装修改后的新服务包  
```hub install deploy/hubserving/clas/```  

- 5、重新启动服务  
```hub serving start -m clas_system```  
