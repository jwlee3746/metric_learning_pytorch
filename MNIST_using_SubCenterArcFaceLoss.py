# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:17:10 2022

@author: Jaewon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

#from torchsummary import summary as summary_

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():   
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# Hyperparameter
BATCH_SIZE = 64
NUM_EPOCHS = 2
EMBEDDING_DIM = 2

img_mean, img_std = (0.1307,), (0.3081,)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(img_mean, img_std)]
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(p = 0.25)
        self.dropout2 = nn.Dropout2d(p = 0.5)
        self.fc1 = nn.Linear(9216, EMBEDDING_DIM)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)     
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        # data.shape : (batchsize, channel, height, width)
        # labels shape : (batchsize)
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        
        embeddings = model(data)
        
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        
        log_interval = 100
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss = {}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss
                )
            )

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=0)
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


# Dataset
train_dataset = datasets.MNIST(root = "./data",
                               train = True,
                               download = True,
                               transform = transform
                               )

test_dataset = datasets.MNIST(root = "./data",
                               train = False,
                               transform = transform
                               )

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True
                                           )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False
                                           )


model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)

### pytorch-metric-learning stuff ###
loss_func = losses.SubCenterArcFaceLoss(num_classes=10, embedding_size=128).to(DEVICE)
loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###

for epoch in range(1, NUM_EPOCHS + 1):
    train(model, loss_func, DEVICE, train_loader, optimizer, loss_optimizer, epoch)
    test(train_dataset, test_dataset, model, accuracy_calculator)