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


def data_loaders(data_dir='./flowers'):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(25),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]) 

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    
    return trainloaders, validloaders, testloaders, train_datasets
    

def build_model(device, hidden_units = 1024, arch='vgg16',):
    if arch == 'vgg16':
        trained_model = models.vgg16(pretrained=True)
        base_value = 25088
    elif arch == 'alexnet':
        trained_model = models.alexnet(pretrained=True)
        base_value = 9216
        
    for param in trained_model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(base_value, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(int(hidden_units/2), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    trained_model.classifier = classifier
    trained_model.to(device)
    return trained_model 

    
def define_opt(trained_model, learning_rate=0.01,):
    
    update_params = []
    for name, param in trained_model.named_parameters():
        if param.requires_grad == True:
            update_params.append(param)
    optimizer = optim.ASGD(update_params, lr = 0.01)
    return optimizer 
    
    
    