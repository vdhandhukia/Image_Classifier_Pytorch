# Imports here
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json
import argparse 
from config_data import *
from config_training import *

# Store command line arguements to configure model inputs
parser = argparse.ArgumentParser(description='Input parameters for image classifier.')

parser.add_argument('data_dir', action ='store', type = str, help = 'path to the folder of flowers ie: ./flowers')
parser.add_argument('--save_dir', type = str, default = './checkpoint.pth', help = 'folder path to save model')
parser.add_argument('--arch', type = str, default = 'alexnet', help = 'Choose pre-trained architecture either vgg16 or alexnet')
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Choose model learning rate (type float)')
parser.add_argument('--hidden_units', type = int, default = 1024, help = 'Choose hidden_units (type int)')
parser.add_argument('--epochs', type = int, default = 15, help = 'Choose epochs number for model (type int)')
parser.add_argument('--gpu', type = str, default = 'gpu', help = 'Choose either gpu or cpu for training')

args = parser.parse_args()

if args.gpu != 'gpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
print(device)    
print(args.save_dir)
print(args.epochs)

trainloaders, validloaders, testloaders, train_datasets = data_loaders(args.data_dir)
model = build_model(device, args.hidden_units, args.arch)
optimizer = define_opt(model, args.learning_rate)

print(optimizer)
print(model)

trained_model = train_model(model, optimizer, trainloaders, validloaders, device, args.epochs) 
print(trained_model)

test_model(trained_model, testloaders, device)
save_model(trained_model, optimizer, args.save_dir, train_datasets, args.epochs, args.hidden_units, args.arch) 


