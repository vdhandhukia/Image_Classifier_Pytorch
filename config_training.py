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


def calc_accuracy(model, dataloaders, device='cpu'):
    test_loss = 0
    accuracy = 0
    model.eval()
    criterion=nn.CrossEntropyLoss()
    
    for images, labels in dataloaders:
        
        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def train_model(trained_model, optimizer, trainloaders, validloaders, device='cpu', epochs=15):
    
    print_every = 32
    steps = 0
    criterion=nn.CrossEntropyLoss()
    
    for e in range(epochs):
        running_loss = 0
        trained_model.train()
        
        for images, labels in iter(trainloaders):
            steps += 1
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            output = trained_model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                trained_model.eval()
                with torch.no_grad():
                    test_loss, accuracy = calc_accuracy(trained_model, validloaders, device)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Valid_Data Loss: {:.3f}.. ".format(test_loss/len(validloaders)),
                    "Valid_data Accuracy: {:.3f}".format(accuracy/len(validloaders)))
            
                running_loss = 0
    
    print("Training is Finished!!")
    return trained_model 


def test_model(trained_model, testloaders, device):
    trained_model.eval()
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        test_loss, accuracy = calc_accuracy(trained_model, testloaders, device)
    print("Test_Data Loss: {:.3f}.. ".format(test_loss/len(testloaders)),
    "Test_data Accuracy: {:.3f}".format(accuracy/len(testloaders)))


def save_model(trained_model, optimizer, save_dir, train_datasets, epochs, hidden_units, arch):
    trained_model.class_to_idx = train_datasets.class_to_idx
    trained_model.cpu()

    torch.save({
        'epoch': epochs,
        'hidden_units': hidden_units,
        'arch': arch,
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': trained_model.class_to_idx, 
        'optimizer_state_dict': optimizer.state_dict()}, 
        save_dir)
    
    print(f"Model has been saved to this path: {save_dir}") 

   










