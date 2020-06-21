import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch import optim
import numpy as np
import time
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import seaborn as sns
import argparse
import json

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',type=str, help="the main flowers data folder containing images e.g.'flowers'")
    parser.add_argument('pathtoimage',type=str, help="the path to an image to classifer, e.g. flowers/valid/1/image_06765.jpg")
    parser.add_argument('checkpoint',type=str, help="the path to the save model checkpoint, e.g. ModelCheckpoint/checkpoint.pth ")
    parser.add_argument('--topk', type=int, default=3, help="returns the top K predictions, default to 3")
    parser.add_argument('--gpu', type=bool, default=True, help="indicates if the model should run on GPU")
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help="the category names file")
    args = parser.parse_args()
    print(args)
    
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'

    # Create a dictionary for the transforms, resize to 255 so the shortest side has length of 255px
    # then center crop the image to 224px.
    # Covert images to numbers with transforms.ToTensor().
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(255),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
    }
    
    model, class_to_idx = load_checkpoint(args.checkpoint)
    
    if(args.gpu):
        # Find the device available to use using torch library
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    probs, classes = predict(args.pathtoimage, model.to(device), args.topk)
    print(probs)
    print(classes)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = image_datasets['train'].classes
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    print(flower_names)
    
# Load checkpoint and rebuild model
def load_checkpoint(filepath):
    
    print(filepath)
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    output = model.forward(image_path)
    output = torch.exp(output)
    
    # Get the top predicted classes and probability
    probs, classes = output.topk(topk, dim=1)
    return probs.item(), classes.item()

if __name__ == "__main__":
    main()
