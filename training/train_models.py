import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from models import *
from loguru import logger
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from torch.nn import DataParallel
import progressbar

models = {
#    "RegNetX_200MF": RegNetX_200MF(),
#    "VGG": VGG('VGG19'),
#    "ResNet18": ResNet18(),
#    "GoogLeNet": GoogLeNet(),
#    "DenseNet121": DenseNet121(),
#    "ResNeXt29_2x64d": ResNeXt29_2x64d(),
#    "MobileNet": MobileNet(),
    #"MobileNetV2": MobileNetV2(),
#    "PreActResNet18": PreActResNet18(),
#    "DPN92": DPN92(),
#    "SENet18": SENet18(),
#    "EfficientNetB0": EfficientNetB0(),
		"DLA_simp": SimpleDLA(),
		"DLA": DLA()
}

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
learning_rate = 0.01
ckpt_path = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Some hyperparams
model_name = 'MobileNet'
net = RegNetX_200MF()
num_epochs = 350
start_epoch = 0
learning_rate = 0.01
train_batch_size = 128
test_batch_size = 100
trainset,trainloader,testset,testloader=None,None,None,None
criterion=None
optimizer=None
scheduler=None

def set_up_params(net):
    global criterion, scheduler, ckpt_path, optimizer
    ckpt_path = "./checkpoint/"
    # Set up optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
    #                    step_size=10.0, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    ckpt_path = ckpt_path + model_name + ".torch"


def set_up_dataset():
    global trainset, trainloader, testset, testloader
    logger.info('==> Preparing data..')
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
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=True, num_workers=2)
    

def train(epoch, net):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        log_interval = 10
        bar_i = 0
        with progressbar.ProgressBar(max_value=len(trainloader)) as bar:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.item()
                _, predicted = outputs.max(1)
                total = total + targets.size(0)
                
                max_logit = outputs.data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                processed = ((batch_idx + 1) * train_batch_size)
                
                accuracy = (100. * correct) / train_batch_size
                progress = (100. * processed) / len(trainset)
                bar_i += 1
                bar.update(bar_i)
                #if (batch_idx + 1) % log_interval == 0: 
                #        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\t', 
                #        epoch, processed, len(trainset), progress, loss.item(), accuracy)


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0.0
    total = 0
    acc = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total

        logger.info('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss.item(), correct, len(testset), acc))  

        if acc > best_acc:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            
            logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}', epoch, loss.item(), ckpt_path )
            torch.save( {'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
												'accuracy': acc}, ckpt_path)
            best_acc = acc

def train_model(start_epoch, net, model_name):
    for epoch in range(start_epoch, start_epoch+num_epochs):
        logger.info("Training {} epoch: {}".format(model_name, epoch))
        train(epoch, net)
        scheduler.step()
        test(epoch, net)



def train_all_models():
    global model_name, best_acc
    
    for model in models.keys():
        model_name = model
        net = DataParallel(models[model]).to(device)
        
        set_up_params(net)
        set_up_dataset()
        train_model(0, net, model_name)
        best_acc = 0

train_all_models()
