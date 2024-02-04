# Transfer-Learning
A repository to fine tune pretrained image clasification models such as Inception, Xception, Resnet-18, 50, 102, GoogleNet, etc. with Pytorch

## Python dependencies
 - pytorch
 - torchvision
 - numpy
 - matplotlib
 - transformers (from huggingface)
 - tinm

## Usage
Data structure should be as follows:
```
data
├── dataset1
│   ├── train
│   │   ├── class1
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ...
│   │   ├── class2
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── test
│   │   ├── class1
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ...
│   │   ├── class2
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ...
│   │   ├── ...
├── dataset2
│   dataset3
│   ...
```


In progress...
