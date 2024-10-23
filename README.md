## Version

Python 3.10

torchvison 0.19.1+cu118  
torch 2.4.1+cu118  

## Requirements
```bash
pip install torch==2.4.1 torchvision==0.19.1 numpy==1.21.2 matplotlib==3.4.3 scikit-learn==1.0.2
git clone https://github.com/Dadaah/v1v1_image_classification_cifar100
```
## How to run
After you have cloned the repository, you can train cifar100 and change seed value by running the script below 
```bash
python /v1v1_image_classification_cifar100/main.py --seed 42 
```
## Implementation Details
| Epoch | Learning Rate | Optimizer | Momentum |
|-------|---------------|-----------|----------|
| 1~30  | 0.1           | SGD       | 0.9      |
| 31~60 | 0.003         | SGD       | 0.9      |
| 61~80 | 0.009         | SGD       | 0.9      |
| 81~100| 0.0008        | SGD       | 0.9      |

## CUDA and GPU Information
CUDA Version: 11.8
GPU: NVIDIA RTX 3090

## Cifar-100 Results

| Network         | Dropout | Preprocess          |   Per Epoch  | Top1 acc(%) | Top5 acc(%) |
|-----------------|---------|---------------------|--------------|-------------|-------------|
| WideResNet 28*20| 0.2     | meanstd             | 11 min 34 sec |    84.37    |    97.01    |
