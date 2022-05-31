# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:38:27 2022

@author: Jaewon
"""
import argparse
import os
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pylab as plt

from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data_utils import TrainDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--data_path', default='D:\GAN\data\celeb\mini_celeba', type=str, help='dataset path')
parser.add_argument('--data_name', default='celeba', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--size', default=64, type=int, help='training images size')
parser.add_argument('--nz', default=128, type=int, help='Latent Vector dim')
parser.add_argument('--out_dim', default=10, type=int, help='ML network output dim')
parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--k', default=1, type=float, help='num training descriminator')
parser.add_argument('--lr', default=1e-4, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.9, type=float, help='Adam optimizer beta2')
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--g_model_name', default='gan', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='', type=str, help='metric learning model name')
parser.add_argument('--margin', default=1, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')
parser.add_argument('--n_threads', type=int, default=16)

if __name__ == '__main__':
    opt = parser.parse_args()
    
    SIZE = opt.size
    out_dim = opt.out_dim
    learning_rate = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
    batch_size = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    num_samples = opt.num_samples
    LOAD_MODEL = opt.load_model
    G_MODEL_NAME = opt.g_model_name
    ML_MODEL_NAME = opt.ml_model_name
    margin = opt.margin
    alpha = opt.alpha
    
    # Result path
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = str(opt.name + '/' + '{}_{}').format(opt.g_model_name, now)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sample_path = output_path           
    
    # Dataset & Dataloader
    train_dataset = TrainDatasetFromFolder(opt.data_path, size=SIZE)  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    
    # Device
    if torch.cuda.is_available():   
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    # Image size
    img_size = (SIZE, SIZE, 3) 

    # Model
    class Generator(nn.Module):
        def __init__(self, nz):
            super(Generator, self).__init__()
            self.nz = nz
            self.main = nn.Sequential(
                nn.Linear(self.nz, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 784),
                nn.Tanh(),
            )
        def forward(self, x):
            return self.main(x).view(-1, 1, 28, 28)
        
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.n_input = 784
            self.main = nn.Sequential(
                nn.Linear(self.n_input, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        def forward(self, x):
            x = x.view(-1, 784)
            return self.main(x)
    
    def save_generator_image(image, path):
        save_image(image, path)
    
    
    def train_discriminator(optimizer, data_real, data_fake):
        b_size = data_real.size(0)
        
        real_label = torch.ones(b_size, 1).to(DEVICE)
        fake_label = torch.zeros(b_size, 1).to(DEVICE)
        
        optimizer.zero_grad()
        output_real = discriminator(data_real)
        loss_real = criterion(output_real, real_label)
        output_fake = discriminator(data_fake)
        loss_fake = criterion(output_fake, fake_label)
        
        loss_real.backward()
        loss_fake.backward()
        optimizer.step()
        return loss_real + loss_fake
    
    def train_generator(optimizer, data_fake):
        b_size = data_fake.size(0)
        
        real_label = torch.ones(b_size, 1).to(DEVICE)
        
        optimizer.zero_grad()
        output = discriminator(data_fake)
        
        loss = criterion(output, real_label)
        loss.backward()
        optimizer.step()
        return loss
    
    ### Traing ###
    generator = Generator(opt.nz).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    optim_g = optim.Adam(generator.parameters(), lr = opt.lr)
    optim_d = optim.Adam(discriminator.parameters(), lr = opt.lr)
    
    criterion = nn.BCELoss()
    
    losses_g = [] 
    losses_d = [] 
    images = [] 
    
    generator.train()
    discriminator.train()
    
    for epoch in range(opt.num_epochs):
        loss_g = 0.0
        loss_d = 0.0
        for idx, data in tqdm(enumerate(train_loader), total=int(len(train_dataset)/train_loader.batch_size)):
            image = data
            image = image.to(DEVICE)
            b_size = len(image)
            for step in range(opt.k):                                
                data_fake = generator(torch.randn(b_size, opt.nz).to(DEVICE)).detach()
                data_real = image
                loss_d += train_discriminator(optim_d, data_real, data_fake)
            data_fake = generator(torch.randn(b_size, opt.nz).to(DEVICE))
            loss_g += train_generator(optim_g, data_fake)
    
        epoch_loss_g = loss_g / idx 
        epoch_loss_d = loss_d / idx 
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)
        print(f"Epoch {epoch} of {opt.num_epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
        
        if epoch % 5 != 0:
            continue
        ###------------------display generated samples--------------------------------------------------    
        generated_img = generator(torch.randn(b_size, opt.nz).to(DEVICE)).cpu().detach()
        generated_img = make_grid(generated_img)
        save_generator_image(generated_img, f"./images/gen_img{epoch}.png")
        images.append(generated_img)
    
    ### Plot loss-graph        
    plt.figure()
    losses_g = [fl.item() for fl in losses_g ]
    plt.plot(losses_g, label='Generator loss')
    losses_d = [f2.item() for f2 in losses_d ]
    plt.plot(losses_d, label='Discriminator Loss')
    plt.legend() 