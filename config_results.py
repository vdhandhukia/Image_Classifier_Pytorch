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


def load_model(checkpoint_file_path, device):
    
    checkpoint = torch.load(checkpoint_file_path)
    hidden_units = checkpoint['hidden_units']
    arch = checkpoint['arch']
    
    if arch == 'vgg16':
        new_model = models.vgg16(pretrained=True)
        base_value = 25088
    elif arch == 'alexnet':
        new_model = models.alexnet(pretrained=True)
        base_value = 9216
    
    for param in new_model.parameters():
        param.requires_grad = False
    
    # Initialize new model
    new_model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(base_value, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(int(hidden_units/2), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    new_model.classifier = classifier
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.to(device)
    
    return new_model


def process_image(image_file_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Resize Image 
    size = (256,256)
    image = Image.open(image_file_path)
    image.thumbnail(size)
    
    #Crop Image
    width, height = image.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    
    #Normalize images
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    #Move Color Channel to first dimension 
    image = image.transpose((2, 0, 1))
    
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, device, category_names, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    # Process image 
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    #Calculate probability 
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model.forward(image.to(device))
    ps = torch.exp(output)
    
    #Take top 5 of the values
    top_five = ps.topk(topk)
    probs_tuple = top_five[0].reshape(-1)
    classes_tuple = top_five[1].reshape(-1)
    
    # Format data to print probability and flower classes
    probs = probs_tuple.cpu().numpy()
    classes_dir_index = classes_tuple.cpu().numpy()
    classes = []
    
    for key, value in model.class_to_idx.items():
        for x in classes_dir_index:
            if value == x:
                classes.append(key) 
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    for key, value in cat_to_name.items():
        for i in range(0, len(classes)):
            if classes[i] in key:
                classes[i] = value
                
    classes = np.asarray(classes, dtype=np.str)
    return probs, classes




