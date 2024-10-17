from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from wide_resnet import Wide_ResNet  #Wide_resnet.py 코드
from torch.autograd import Variable
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

seed = 42  # Set seed value for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA availability
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if use_cuda else "cpu")
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

train_losses = []  # Save training loss for each epoch
test_losses = []   # Save test loss for each epoch


# Data Preparation
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
])

# CIFAR-100 dataset
print("| Preparing CIFAR-100 dataset...")
sys.stdout.write("| ")
full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

train_indices, val_indices = train_test_split(
    list(range(len(full_trainset))),
    test_size=0.2,
    stratify=full_trainset.targets,
    random_state=42
)

trainset = torch.utils.data.Subset(full_trainset, train_indices)
valset = torch.utils.data.Subset(full_trainset, val_indices)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

num_classes = 100

# Initialize and return network
def getNetwork():
    net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
    return net, file_name

# Model setup
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork()
    checkpoint = torch.load('./checkpoint/cifar100/' + file_name + '.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building Wide-ResNet...')
    net, file_name = getNetwork()

if use_cuda:
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    print(f'\n=> Training Epoch #{epoch}, LR={cf.learning_rate(args.lr, epoch):.4f}')
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sys.stdout.write(f'\r| Epoch [{epoch}/{num_epochs}] Iter[{batch_idx+1}/{len(trainloader)}]\tLoss: {loss.item():.4f} Acc: {100.*correct/total:.3f}%')
        sys.stdout.flush()

    train_loss = train_loss / len(trainloader)
    train_losses.append(train_loss)

# Validation/Test
def test(epoch, loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(loader)
    test_losses.append(test_loss)
    acc = 100.*correct/total
    print(f"\n| Test Epoch #{epoch}\tLoss: {test_loss:.4f} Acc: {acc:.2f}%")
    return acc

# Training phase
print('\n[Phase 3] : Training model')
print(f'| Training Epochs = {num_epochs}')
print(f'| Initial Learning Rate = {args.lr}')
print(f'| Optimizer = {optim_type}')

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    
    train(epoch)
    acc = test(epoch, valloader)
    
    # Save best model
    if acc > best_acc:
        print(f'| Saving Best model... Acc: {acc:.2f}%')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/cifar100/'
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + file_name + '.t7')
        best_acc = acc
    
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print(f'| Elapsed time: {cf.get_hms(elapsed_time)}')

# Final test
print('\n[Phase 4] : Testing model')
final_acc = test(start_epoch + num_epochs - 1, testloader)
print(f'* Test results with Best Model: Acc: {final_acc:.2f}%')

# Plot loss
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Test Loss')
plt.show()
