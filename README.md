# Binary Neural Networks (BNN)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![PyPI](https://img.shields.io/pypi/v/bnn.svg?style=flat)](https://pypi.org/project/bnn/)

BNN is a Pytorch based library that facilitates the binarization (i.e. 1 bit quantization) of neural networks.

## Installation

### Requirements

* Python 3.7+
* PyTorch (>=1.8)

The easiest way to install the package is using pip or conda. Alternatively you can install the package from source.

| **Using pip**                | **Using conda**                            |
|------------------------------|--------------------------------------------|
| `pip install bnn`            | `conda install -c 1adrianb bnn`            |

## Why network binarization?

Network binarization is the most extreme case of quantization restricting the input features and/or weights to two states only {-1,1}. Such hardware friendly representation can reduce the size of a float32 layer by **x32** times via bitpacking. Similarly, on modern x64 CPUs the operations can be executed up to **x64** faster via SIMD. Note that in order to take advantage at runtime of such speed-ups a hardware-friendly implementation is required which the current repo doesn't include currently.

## Quick start

In order to facilitate common chaining operation that typically occur when binarizing neural networks we provide an easy mechanism to achieve this via a set of yaml configuration files (herein called recipes). An example of such file can be found in the recipes folder.

Note that the examples provided bellow are simply intended to showcase the API are not necessarily the optimal configurations. For a more detailed behaviour of the available functions please check the corresponding documentation and research papers. The examples folder provides a full working example.

### **1. Explicit usage**

Similarly with the pytorch quantization module we can define a binarization configuration  that will contains the binarization strategies(modules) used. Once defined, the `prepare_binary_model` function will propagate them to all nodes and then swap the modules with the fake binarized ones.
Alternatively, the user can define manually, at network creation time, the bconfig for each layer and then call then `convert` function to swap the modules appropriately.

```python
import torch
import torchvision.models as models

from bnn import BConfig, prepare_binary_model
# Import a few examples of quantizers
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer

# Create your desire model (note the default R18 may be suboptimal)
# additional binarization friendly models are available in bnn.models
model = models.resnet18(pretrained=False)

# Define the binarization configuration and assign it to the model
model.bconfig = BConfig(
    activation_pre_process = BasicInputBinarizer,
    activation_post_process = BasicScaleBinarizer,
    # optionally, one can pass certain custom variables
    weight_pre_process = XNORWeightBinarizer.with_args(center_weights=True)
)
# Convert the model appropiately, propagating the changes from parent node to leafs
# The custom_config_layers_name syntax will perform a match based on the layer name, setting a custom quantization function.
bmodel = prepare_binary_model(model, bconfig, custom_config_layers_name=['conv1' : BConfig()])

# You can also ignore certain layers using the ignore_layers_name. 
# To pass regex expression, frame them between $ symbols, i.e.: $expression$.

```

### **2. Using binarization recepies**

```python
import torch
import torchvision.models as models

# Import the recepies consumer enginer
from bnn.executor.engine import BinaryChef

# Create your desire model (note the default R18 may be suboptimal)
model = models.resnet18(pretrained=False)
chef = BinaryChef('../recepies/xnor-net.yaml')

# Repeat the training procedure using the steps define in the config file
for _ in range(len(chef)):
    # Convert the model according to the recepie
    model = chef.next(model)

    ### Run here your training logich for N epochs
```

### **3. Implementing a custom weight binarizer**

Implementing custom operations is a straightforward process. You can simply define your new classpython register class to a given module:

```python
import torch.nn as nn
import torch.nn.functional as F

class CustomOutputBinarizer(nn.Module):
    def __init__(self):
        super(CustomOutputBinarizer, self).__init__()
        
    def forward(self, x_after, x_before):
        # scale binarizer takes a list of input containg [conv_output and conv_input]
        return F.normalize(x_after, p=2) # operate on the conv_output

class CustomInputBinarizer(nn.Module):
    def __init__(self):
        super(CustomInputBinarizer, self).__init__()
        
    def forward(self, x):
        # dummy example of using sign instead of tanh
        return torch.tanh(x) # operate on the conv_output

# apply the custom functions into the binarization model
model.bconfig = BConfig(
    activation_pre_process = CustomInputBinarizer,
    activation_post_process = CustomOutputBinarizer,
    weight_pre_process = nn.Identity # this will keep the weights real
)

```

### **4. Using adapted architecures**

While existing of the shelves modules can be used directly, binarizing them may prove problematic.
The `bnn.models` implement a few popular choices:

  1. Hierarchical Block - *Hierarchical binary CNNs for landmark localization with limited resources, A. Bulat, G. Tzimiropoulos, IEEE TPAMI 2020 (<https://arxiv.org/abs/1808.04803>).*
  2. Residual layers with pre-activation - *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks, M. Rastegari, V. Ordonez, J. Redmond, A. Farhadi, ECCV 2016 (<https://arxiv.org/abs/1603.05279>).*
  3. PReLU/Leaky ReLU instead of ReLU - *Improved training of binary networks for human pose estimation and image recognition, A. Bulat, G. Tzimiropoulos, J. Kossaifi, M. Pantic, arXiv 2019 (<https://arxiv.org/abs/1904.05868>).*
  4. Efficient ResNet stem - *daBNN: A Super Fast Inference Framework for Binary Neural Networks on ARM devices, J. Zhang, Y. Pan, T. Yao, H. Zhao, T. Mei, ACMMM 2019 (<https://arxiv.org/abs/1908.05858>).*
  5. BATS NAS - *BATS: Binary ArchitecTure Search, A. Bulat, B. Martinez, G. Tzimiropoulos, ECCV 2020 (<https://arxiv.org/abs/2003.01711>)*

Note that they are implemented based on the descriptions provided in the original paper

### **5. Counting FLOPs and BOPs (binary operations)**

This aspect makes usage of our _pthflops_ package. For instalation instructions please visit [https://github.com/1adrianb/pytorch-estimate-flops](https://github.com/1adrianb/pytorch-estimate-flops).

```python
from pthflops import count_ops

device = 'cuda:0'
inp = torch.rand(1,3,224,224).to(device)

all_ops, all_data = count_ops(model, inp)

flops, bops = 0, 0
for op_name, ops_count in all_data.items():
    if 'Conv2d' in op_name and 'onnx::' not in op_name:
        bops += ops_count
    else:
        flops += ops_count

print('Total number of FLOPs: {}', flops)
print('Total number of BOPs: {}', bops)

```

## Contributing

All contributions are highly welcomed. Feel free to self-assign yourself to existing issues, or open a new pull request if you would like to add a features. For new features, opening a issue for having a prior discussion is probably the best course of action.

## Citation

This code was developed during my PhD done at University of Nottingham and is released in support of my thesis.
If you found this package helpfull, please cite:

```lang-latex
@inproceedings{bulat2017binarized,
  title={Binarized convolutional landmark localizers for human pose estimation and face alignment with limited resources},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3706--3714},
  year={2017}
}
```
