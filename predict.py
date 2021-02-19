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
from config_results import * 

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Store command line arguements to configure model inputs
parser = argparse.ArgumentParser(description='Input parameters for classifier prediction results.')

parser.add_argument('input', type = str, default = './flowers/test/10/image_07090.jpg', help = 'path to the image of a flower')
parser.add_argument('checkpoint', type = str, default = './checkpoint.pth', help = 'Call folder path that saved training model param')
parser.add_argument('--top_k', type = int, default = 5, help = 'Gives top number of highest probabilities/matches (type int)')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Call folder path that maps categories to real names')
parser.add_argument('--gpu', type = str, default = 'gpu', help = 'Choose either gpu or cpu for device')
#parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Choose pre-trained architecture that matches saved model')

args = parser.parse_args()

if args.gpu != 'gpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

print(args.input)
print(args.checkpoint)
new_model = load_model(args.checkpoint, device)
probs, classes = predict(args.input, new_model, device, args.category_names, args.top_k)
print(probs)
print(classes)

print(f"The top {args.top_k} probabilities of the selected image are: ")
i = 0
for i in range(0, len(probs)):
    print(f"{classes[i]}, {probs[i]}")








