# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:38:27 2022

@author: Jaewon
"""
import argparse
import os, time, sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pickle
from math import log10,exp,sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pylab as plt

from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

from data_utils import TrainDatasetFromFolder
from model_gan import generator, discriminator

from data_utils import is_image_file, train_transform, TrainDatasetFromFolder, \
                        generate_image, gen_rand_noise, remove_module_str_in_state_dict, requires_grad

parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--data_path', default='D:\GAN\data\celeb\mini_celeba', type=str, help='dataset path')
parser.add_argument('--data_name', default='celeba', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--size', default=64, type=int, help='training images size')
parser.add_argument('--nz', default=128, type=int, help='Latent Vector dim')
parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--k', default=1, type=float, help='num training descriminator')
parser.add_argument('--lr', default=1e-2, type=float, help='train learning rate')
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
    log_path= str(output_path + '/logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    sample_path = output_path           
    
    # jupyter-tensorboard writer
    writer = SummaryWriter(log_path)    
    
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
    
    # network
    netG = generator(128)
    netD = discriminator(128)
    netG.weight_init(mean=0.0, std=0.02)
    netD.weight_init(mean=0.0, std=0.02)
    netG.to(DEVICE) 
    netD.to(DEVICE) 
    
    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss().to(DEVICE) 
    
    # Adam optimizer
    G_optimizer = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Train
    for epoch in range(1, NUM_EPOCHS + 1):
            
        train_bar = tqdm(train_loader) 
    
        netG.train()
        netD.train()    
    
        for target in train_bar:            
            real_img = Variable(target).to(DEVICE)           
                
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            ############################ 
            # train discriminator D
            ############################                                        
            requires_grad(netG, False)
            requires_grad(netD, True)
    
            y_real_ = torch.ones(batch_size).to(DEVICE) 
            y_fake_ = torch.zeros(batch_size).to(DEVICE) 
                                        
            D_real_result = netD(real_img).squeeze()
            D_real_loss = BCE_loss(D_real_result, y_real_)
    
            z_ = torch.randn((batch_size, 128)).view(-1, 128, 1, 1).to(DEVICE)      
            z_.requires_grad_(True)
            G_result = netG(z_)
    
            D_fake_result = netD(G_result).squeeze()
            D_fake_loss = BCE_loss(D_fake_result, y_fake_)
            D_fake_score = D_fake_result.data.mean()
    
            D_train_loss = D_real_loss + D_fake_loss
    
            netD.zero_grad()
            D_train_loss.backward()
            D_optimizer.step()
    
    
            ############################ 
            # train generator G
            ############################         
            requires_grad(netG, True)
            requires_grad(netD, False)
    
            G_result = netG(z_)
            D_result = netD(G_result).squeeze()
            
            G_train_loss = BCE_loss(D_result, y_real_)
            
            netG.zero_grad()  
            G_train_loss.backward()
            G_optimizer.step()   
            
            train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
        
        print(' D_train_loss:', D_train_loss.item(), ' G_train_loss:', G_train_loss.item())
        writer.add_scalars("loss/train", {"d_loss": D_train_loss, "g_loss": G_train_loss}, epoch)
        #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
        #writer.add_scalar("g_loss/train", g_loss, epoch)
        
        if epoch % 2 != 0:
            continue    

###------------------display generated samples--------------------------------------------------        
        fixed_noise = gen_rand_noise(num_samples).to(DEVICE) 
        gen_images = generate_image(netG, dim=SIZE, batch_size=num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)             
        
        if epoch % 10 != 0:
            continue    
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netD.state_dict(), str(output_path  +'/' + "d_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_latest.pt"))
        torch.save(netD.state_dict(), str(output_path  +'/' + "d_model_latest.pt"))
    
    writer.close()