# Itrerative Feature Merging(IFM)

## Official implementation of the "GOING BEYOND NEURAL NETWORK FEATURE SIMILARITY: THE NETWORK FEATURE COMPLEXITY AND ITS INTERPRETATION USING CATEGORY THEORY" ICLR2024

## Introduction
In this paper, we expand the concept of equivalent feature and provide the definition of what we call functionally
equivalent features through the lens of category theory. Using this definition, we derive a more intrinsic metric for the
layer-wise feature complexity regarding the redundancy of features learned by a
neural network at each layer. We propose an efficient algorithm named
Iterative Feature Merging (IFM) to measure the feature complexity. IFM merges similar channels of each layer which reduces computational cost with little performance loss.

## The IFM algorithm
![An illustration of our proposed functionally equivalent features and feature complexity. Two different models learn functionally equivalent features if the features and outputs of the two models are equivalent under certain invertible linear transformations. Feature complexity (layer-wise) is the dimensionality of the most compact representation of the feature at a certain layer. In order to retrieve the most compact version, we propose iterative feature merging (IFM).](overview.png)
Two different models learn functionally equivalent features if the features and outputs of the two models are equivalent under certain invertible linear transformations. Feature complexity (layer-wise) is the dimensionality of the most compact representation of the feature at a certain layer. In order to retrieve the most compact version, we propose iterative feature merging (IFM). The IFM algorithm is simple such that we repeatedly merge similar channels until the minimum difference between two channels are above the threshold.

## Install Requirements
The codebase is built and tested with Python 3.9.5. To install required packages:

```pip install -r requrements.txt```

## Run IFM algorithm

### Train a model or download a pretrained checkpoint
For models on ImageNet, we use the checkpoint provided in [torchvision][torchvision_link]. For CIFAR10, we provide code to train models from scratch. To train a model on CIFAR10, please use

```python train_model.py --cfg ./cfg/${your model name}_training.yml --prefix ${prefix name}```

### Apply IFM on the checkpoint of a trained model
The IFM algorithm only have one hyper-parameter $\beta$ controlling the threshold, which should be a decimal between 0 and 1. In this repo, we use ``threshold`` to determine the $\beta$. For more flexibility, we add a hyper-parameter ``max_ratio``, which limits the maximum ratio of channels being merged. For example, ``max_ratio=0.1`` means that less than 10% channels will be reduced (merged).  
To apply IFM on a checkpoint and test the model, please run

```python ${model name}_{dataset}_self_merge.py --model_param ${path to the checkpoint} --threshold ${\beta in IFM, controlling the threshold to stop} --max_ratio ${default as 1.0}```

We also provide bash script to run the code, please refer to ``${model name}_{dataset}_self_merge.sh`` and change the model_param to the path of the target checkpoint.

[torchvision_link]: https://pytorch.org/vision/stable/models.html