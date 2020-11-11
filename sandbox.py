# imports
from exceptions.Exceptions import ModelNotFoundException
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from loguru import logger
import torchvision
import torchvision.transforms as transforms
from models import *
import os
import argparse



class Sandbox():

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainloader = None
    testloader = None
    net = None
    learning_rate = 0.01
    ckpt_path = "./checkpoint/"

    models = {
        "RegNetX_200MF": RegNetX_200MF(),
        "VGG": VGG('VGG19'),
        "ResNet18": ResNet18(),
        "PreActResNet18": PreActResNet18(),
        "GoogLeNet": GoogLeNet(),
        "DenseNet121": DenseNet121(),
        "ResNeXt29_2x64d": ResNeXt29_2x64d(),
        "MobileNet": MobileNet(),
        "MobileNetV2": MobileNetV2(),
        "DPN92": DPN92(),
        "ShuffleNetG2": ShuffleNetG2(),
        "SENet18": SENet18(),
        "ShuffleNetV2": ShuffleNetV2(),
        "EfficientNetB0": EfficientNetB0()
    }

    def __init__(self, hparams):
        # Set up optimizer
        self.learning_rate = hparams.learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
                            momentum=0.9, weight_decay=5e-4)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if hparams.model in self.models.keys():
            self.net = self.models.get(hparams.model)
        else:
            raise ModelNotFoundException
        
        self.ckpt_path = self.ckpt_path + hparams.model + ".torch"
        self.num_epochs = hparams.num_epochs
        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

def prepare_dataset(self):
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

    self.trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    self.trainloader = torch.utils.data.DataLoader(
        self.trainset, batch_size=128, shuffle=True, num_workers=2)

    self.testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    self.testloader = torch.utils.data.DataLoader(
        self.testset, batch_size=100, shuffle=True, num_workers=2)

    
    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        log_interval = 10
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss = train_loss + loss.item()
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()
            processed = ((batch_idx + 1) * 128)
            accuracy = (100.0 * correct) / 128
            progress = (100. * processed) / len(self.trainset)

            if (batch_idx + 1) % log_interval == 0: 
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\t', 
                    epoch, processed, len(self.trainset), progress, train_loss, accuracy)


    def test(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        acc = 0

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total

            logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.testset), acc))  

            if acc > self.best_acc:
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, './checkpoint' )
                torch.save( {'epoch': epoch,
                            'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.loss}, self.ckpt_path)
                best_acc = acc
    
    def train_model(start_epoch):
        for epoch in range(start_epoch, start_epoch+self.num_epochs):
            train(epoch)
            test(epoch)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        type=str,
                        default="RegNetX_200MF",
                        help="Which DCNN model to use.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,
                        help="Learning rate for gradient descent")
    
    parser.add_argument("--num_epochs",
                        type=int,
                        default=50,
                        help="How many epochs to run model for.")

    # TODO: Implement loading of previously trained models
    sandbox = Sandbox(parser.parse_args())
    sandbox.train_model(0)