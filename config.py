import math

start_epoch = 1
num_epochs = 100
batch_size = 128
optim_type = 'sgd'

mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 80):
        optim_factor = 5
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 30):
        optim_factor = 1

    return init*math.pow(0.3, optim_factor)# 변경 후 

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
