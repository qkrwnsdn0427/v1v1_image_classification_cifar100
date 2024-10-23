## Requirements

```bash
pip install torchvision
pip install matplotlib
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


## Cifar-100 Results

| Network         | Dropout | Preprocess          |   Per Epoch  | Top1 acc(%) | Top5 acc(%) |
|-----------------|---------|---------------------|--------------|-------------|-------------|
| WideResNet 28*20| 0.2     | meanstd             | 6 min 40 sec |    83.52    |    96.06    |
