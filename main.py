from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime


from wide_resnet import *  #Wide_resnet.py
from torch.autograd import Variable
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split  # Stratified Sampling을 위한 라이브러리 추가
import matplotlib.pyplot as plt

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--seed', default=42, type=float, help='seed')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = cifar100')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()
seed = 42  # 원하는 seed 값으로 변경 가능
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 경우 추가
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 스크립트 초반에 device를 정의
device = torch.device("cuda" if use_cuda else "cpu")
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

train_losses = []  # 각 epoch의 training loss를 저장할 리스트
test_losses = []   # 각 epoch의 test loss를 저장할 리스트


# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),   
    # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),  # 기하학적 변환
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),  # 확대/축소 변환만 적용
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


# if(args.dataset == 'cifar100'):
print("| Preparing CIFAR-100 dataset...")
sys.stdout.write("| ")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

# ###  validation 학습 4만장 검증 1만장용
# ### 학습 데이터 4만장, 검증 데이터 1만장으로 나누기 (계층적 샘플링 적용) ###
# train_indices, val_indices = train_test_split(
#     list(range(len(full_trainset))), 
#     test_size=0.2,  # 4만장: 학습, 1만장: 검증 (20% 검증)
#     stratify=full_trainset.targets,  # 각 클래스 비율을 유지하여 나눔
#     random_state=42
# )

# trainset = torch.utils.data.Subset(full_trainset, train_indices)  # 학습 데이터 (4만장)
# valset = torch.utils.data.Subset(full_trainset, val_indices)      # 검증 데이터 (1만장)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,worker_init_fn = seed_worker)
# valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# CIFAR-100 Superclass와 각 클래스의 매핑 (index 기준)
superclass_mapping = {
    0: [4, 30, 55, 72, 95],      # aquatic mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food containers
    4: [0, 51, 53, 57, 83],      # fruit and vegetables
    5: [22, 39, 40, 86, 87],     # household electrical devices
    6: [5, 20, 25, 84, 94],      # household furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large carnivores
    9: [12, 17, 37, 68, 76],     # large man-made outdoor things
    10: [23, 33, 49, 60, 71],    # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],    # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],    # medium-sized mammals
    13: [26, 45, 77, 79, 99],    # non-insect invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small mammals
    17: [47, 52, 56, 59, 96],     # trees
    18: [8, 13, 48, 58, 90],    # vehicles 1
    19: [41, 69, 81, 85, 89]     # vehicles 2
}

# 각 클래스가 속한 superclass를 저장한 리스트 생성
class_to_superclass = [None] * 100
for super_idx, class_indices in superclass_mapping.items():
    for class_idx in class_indices:
        class_to_superclass[class_idx] = super_idx          

# Return network & file name
def getNetwork(args):
    net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
    file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)

    return net, file_name

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# CutMix 함수
def cutmix_data(x, y, beta=1.0):
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

# 랜덤 박스 생성 함수
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
     # 옵티마이저 선택
    if optim_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    elif optim_type == 'sgd_momentum':
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif optim_type == 'nadam':
        optimizer = optim.NAdam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")


    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, beta=1.0)
        outputs = net(inputs)               # Forward Propagation
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)  # Loss
        loss.backward()# Backward Propagation
        optimizer.step()
        

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

    # epoch의 평균 training loss 저장
    train_loss = train_loss / len(trainloader)
    train_losses.append(train_loss)
        
def map_to_superclass(labels, class_to_superclass):
    return [class_to_superclass[label] for label in labels]
        
def evaluate_superclass_and_general_accuracy(model, test_loader, device, class_to_superclass, k=5):
    model.eval()  # 평가 모드
    correct_top1_general = 0
    correct_top5_general = 0
    correct_top1_superclass = 0
    correct_top5_superclass = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, top5_pred = torch.topk(outputs, k, dim=1)
            _, predicted_top1 = torch.max(outputs, 1)  # 일반 Top-1 예측

            # 일반 Top-1 정확도 계산
            correct_top1_general += (predicted_top1 == labels).sum().item()
            
            # 일반 Top-5 정확도 계산
            correct_top5_general += sum([labels[i] in top5_pred[i] for i in range(labels.size(0))])
            
            # 실제 라벨을 슈퍼클래스로 변환
            true_superclass = torch.tensor(map_to_superclass(labels.cpu().numpy(), class_to_superclass)).to(device)
            pred_superclasses_top1 = torch.tensor(map_to_superclass(predicted_top1.cpu().numpy(), class_to_superclass)).to(device)

            # 슈퍼클래스 Top-1 정확도
            correct_top1_superclass += (pred_superclasses_top1 == true_superclass).sum().item()

            # 슈퍼클래스 Top-5 정확도
            pred_superclasses_top5 = [torch.tensor(map_to_superclass(pred.cpu().numpy(), class_to_superclass)).to(device) for pred in top5_pred]
            for i in range(labels.size(0)):
                true_label_superclass = true_superclass[i].item()
                if true_label_superclass in pred_superclasses_top5[i].cpu().numpy():
                    correct_top5_superclass += 1

            total += labels.size(0)

     # 정확도 계산
    top1_accuracy_general = 100 * correct_top1_general / total
    top5_accuracy_general = 100 * correct_top5_general / total
    top1_accuracy_superclass = 100 * correct_top1_superclass / total
    top5_accuracy_superclass = 100 * correct_top5_superclass / total
    
    return top1_accuracy_general, top5_accuracy_general, top1_accuracy_superclass, top5_accuracy_superclass

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    correct_top1_general = 0
    correct_top5_general = 0
    correct_top1_superclass = 0
    correct_top5_superclass = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            # 일반 Top-1 및 Top-5 정확도 계산
            _, predicted_top1 = torch.max(outputs.data, 1)
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_top1_general += predicted_top1.eq(targets.data).cpu().sum()
            correct_top5_general += sum([targets[i] in top5_pred[i] for i in range(targets.size(0))])


            # 슈퍼클래스 정확도 계산
            true_superclass = torch.tensor(map_to_superclass(targets.cpu().numpy(), class_to_superclass)).to(device)
            pred_superclasses_top1 = torch.tensor(map_to_superclass(predicted_top1.cpu().numpy(), class_to_superclass)).to(device)

            correct_top1_superclass += (pred_superclasses_top1 == true_superclass).sum().item()

            pred_superclasses_top5 = [torch.tensor(map_to_superclass(pred.cpu().numpy(), class_to_superclass)).to(device) for pred in top5_pred]
            for i in range(targets.size(0)):
                true_label_superclass = true_superclass[i].item()
                if true_label_superclass in pred_superclasses_top5[i].cpu().numpy():
                    correct_top5_superclass += 1

                # epoch의 평균 test loss 저장
        test_loss = test_loss / len(testloader)
        test_losses.append(test_loss)
        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        
        # 일반 정확도
        acc_top1_general = 100. * correct_top1_general / total
        acc_top5_general = 100. * correct_top5_general / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@5: %.2f%%" %(epoch, loss.item(), acc_top5_general))
        # 슈퍼클래스 정확도
        acc_top1_superclass = 100. * correct_top1_superclass / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f SuperClassAcc@1: %.2f%%" %(epoch, loss.item(), acc_top1_superclass))
        acc_top5_superclass = 100. * correct_top5_superclass / total

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            best_acc = acc
        
        return acc_top1_general, acc_top5_general, acc_top1_superclass, acc_top5_superclass

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        # 모델을 불러올 때 수정


    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    correct_top1_general = 0
    correct_top5_general = 0
    correct_top1_superclass = 0
    correct_top5_superclass = 0
    top1_acc, top5_acc, top1_superclass_acc, top5_superclass_acc = test(0)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            # General Top-1 및 Top-5 정확도 계산
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            _, predicted_top1 = torch.max(outputs, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_top1_general += predicted_top1.eq(targets.data).cpu().sum()
            correct_top5_general += sum([targets[i] in top5_pred[i] for i in range(targets.size(0))])

            # Superclass 정확도 계산
            true_superclass = torch.tensor(map_to_superclass(targets.cpu().numpy(), class_to_superclass)).to(device)
            pred_superclasses_top1 = torch.tensor(map_to_superclass(predicted_top1.cpu().numpy(), class_to_superclass)).to(device)

            correct_top1_superclass += (pred_superclasses_top1 == true_superclass).sum().item()

            pred_superclasses_top5 = [torch.tensor(map_to_superclass(pred.cpu().numpy(), class_to_superclass)).to(device) for pred in top5_pred]
            for i in range(targets.size(0)):
                true_label_superclass = true_superclass[i].item()
                if true_label_superclass in pred_superclasses_top5[i].cpu().numpy():
                    correct_top5_superclass += 1

        # Test 결과 출력
        acc_top1_general = 100. * correct_top1_general / total
        acc_top5_general = 100. * correct_top5_general / total
        acc_top1_superclass = 100. * correct_top1_superclass / total
        acc_top5_superclass = 100. * correct_top5_superclass / total

        acc = 100.*correct/total
        print(f"-----------------Best Model----------------------")
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
        # Test 결과 출력
        print(f"* Test results: General Top-1 Accuracy = {acc_top1_general:.2f}%")
        print(f"* General Top-5 Accuracy = {acc_top5_general:.2f}%")
        print(f"* Superclass Top-1 Accuracy = {acc_top1_superclass:.2f}%")
        print(f"* Superclass Top-5 Accuracy = {acc_top5_superclass:.2f}%")
        print(f"---------------------------------------------------------")
        print(f"-----------------Last Epoch Model----------------------")
        print(f"* Test results: General Top-1 Accuracy = {top1_acc:.2f}%")
        print(f"* General Top-5 Accuracy = {top5_acc:.2f}%")
        print(f"* Superclass Top-1 Accuracy = {top1_superclass_acc:.2f}%")
        print(f"* Superclass Top-5 Accuracy = {top5_superclass_acc:.2f}%")

    sys.exit(0)

# # 검증을 위한 함수 (valloader 사용)
# def validate(epoch, last_epoch=False):
#     return test(epoch, last_epoch, loader=valloader)

# # 테스트를 위한 함수 (testloader 사용)
# def test_final(epoch, last_epoch=False):
#     return test(epoch, last_epoch, loader=testloader)

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)
    # # 매 에포크마다 검증 단계 실행
    # validate(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')

# 최종 테스트 실행
top1_acc, top5_acc, top1_superclass_acc, top5_superclass_acc = test(cf.num_epochs)
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

# # 최고 성능 모델로 테스트 진행
# top1_acc, top5_acc, top1_superclass_acc, top5_superclass_acc = test_final(cf.num_epochs - 1, last_epoch=True) #, loader=testloader
# print('* Test results with Best Model : Acc@1 = %.2f%%' %(best_acc))

# 최종 결과 출력
print(f"Final General Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Final General Top-5 Accuracy: {top5_acc:.2f}%")
print(f"Final Superclass Top-1 Accuracy: {top1_superclass_acc:.2f}%")
print(f"Final Superclass Top-5 Accuracy: {top5_superclass_acc:.2f}%")

for epoch in range(len(train_losses)):
    print(f"Epoch {epoch+1}: Training Loss = {train_losses[epoch]:.4f}, Test Loss = {test_losses[epoch]:.4f}")

# loss 시각화
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Test Loss')
plt.show()
