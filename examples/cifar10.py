import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from bnn.ops import  (
    BasicInputBinarizer,
    XNORWeightBinarizer
)
from bnn import BConfig, prepare_binary_model
from  bnn.models.resnet import resnet18

from .utils import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--print_freq', type=int, default=100,
                    help='logs printing frequency')
parser.add_argument('--out_dir', type=str, default='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = resnet18()

# Binarize
print('==> Preparing the model for binarization')
bconfig = BConfig(
            activation_pre_process = BasicInputBinarizer,
            activation_post_process = nn.Identity,
            weight_pre_process = XNORWeightBinarizer
        )
# first and last layer will be kept FP32
model = prepare_binary_model(net, bconfig, custom_config_layers_name={'conv1': BConfig(), 'fc': BConfig()})
print(model)

net = net.to(device)
if 'cuda' in device:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)

# Training
def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    print('\nTrain Epoch: %d' % epoch)
    net.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, = accuracy(outputs, targets)

        top1.update(acc1.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)


def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, data_time, losses, top1],
        prefix="Test Epoch: [{}]".format(epoch))

    global best_acc
    net.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets)

            top1.update(acc1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    print('Current acc: {}, best acc: {}'.format(acc, best_acc))


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

