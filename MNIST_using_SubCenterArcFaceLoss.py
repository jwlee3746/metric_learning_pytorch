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
import torchvision

if torch.cuda.is_available():   
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# Hyperparameter
BATCH_SIZE = 64
NUM_EPOCHS = 2
EMBEDDING_DIM = 128

# Model, methods
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
        
        ### plot scatter image for 1 batch###
        if EMBEDDING_DIM <= 3 and batch_idx == 0:
            temp_data, temp_label = data, labels
            embeddings = model(temp_data).cpu().detach().numpy()
            temp_label = temp_label.cpu().detach().numpy()
            
            figsize = 10
            
            if EMBEDDING_DIM == 2:
                plt.figure(figsize=(figsize, figsize))
                plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                            cmap='rainbow', c=temp_label, alpha=0.7, s=15)
                
            elif EMBEDDING_DIM == 3:
                fig = plt.figure(figsize=(figsize, figsize))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                            cmap='rainbow', c=temp_label, alpha=0.7, s=10)
            
            plt.show()
        ### plot scatter image for batch-1### 
        
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

### Plot n-dim(n : 2-3) data scatter after train ###
def plot_after_train(model, n_to_show = 1000, train_or_test = "train"):
    grid_size = 15
    figsize = 10
    
    if train_or_test == "train":
        example_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                   batch_size=n_to_show,
                                                   shuffle=True
                                                   )
    elif train_or_test == "test":
        example_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                   batch_size=n_to_show,
                                                   shuffle=False
                                                   )
    
    for batch_idx, (data, labels) in enumerate(example_loader):
        data, labels = data.to(DEVICE), labels
        embeddings = model(data).cpu().detach().numpy()
        
        min_x = min(embeddings[:, 0]).item()
        max_x = max(embeddings[:, 0]).item()
        min_y = min(embeddings[:, 1]).item()
        max_y = max(embeddings[:, 1]).item()
        
        if EMBEDDING_DIM == 2:
            plt.figure(figsize=(figsize, figsize))
            plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                        cmap='rainbow', c=labels, alpha=0.7, s=15)
            
        elif EMBEDDING_DIM == 3:
            min_z = min(embeddings[:, 2]).item()
            max_z = max(embeddings[:, 2]).item()
        
            fig = plt.figure(figsize=(figsize, figsize))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                        cmap='rainbow', c=labels, alpha=0.7, s=10)
        
        plt.show()    
        break

# Dataset

img_mean, img_std = (0.1307,), (0.3081,)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(img_mean, img_std)]
)

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
loss_func = losses.SubCenterArcFaceLoss(num_classes=10, embedding_size=EMBEDDING_DIM).to(DEVICE)
loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###

for epoch in range(1, NUM_EPOCHS + 1):
    train(model, loss_func, DEVICE, train_loader, optimizer, loss_optimizer, epoch)
    test(train_dataset, test_dataset, model, accuracy_calculator)

if EMBEDDING_DIM <= 3:
    plot_after_train(model, train_or_test="train")
    plot_after_train(model, train_or_test="test")

### View some sample outliers
train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
outliers, dominant_centers = loss_func.get_outliers(train_embeddings, train_labels.squeeze(1), return_dominant_centers=True)
print(f"\nThere are {len(outliers)} outliers")

inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(img_mean, img_std)], std=[1 / s for s in img_std]
)


def imshow(img, figsize=(8, 4)):
    img = inv_normalize(img)
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow_many(dataset, outliers, n=32):
    imgs = [
        dataset[outliers[i]][0]
        for i in np.random.choice(
            len(outliers), size=min(n, len(outliers)), replace=False
        )
    ]
    imshow(torchvision.utils.make_grid(imgs))

imshow_many(train_dataset, outliers)
### View some sample outliers