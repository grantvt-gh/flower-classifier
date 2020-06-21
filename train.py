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
import torch.nn.functional as F
import argparse
import json

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',type=str, help="the main flowers data folder containing images e.g.'flowers'")
    parser.add_argument('--arch',default="densenet161",type=str, help="the CNN model to use, defaults to densenet161")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="the optimizer learning rate, defaults to 0.001")
    parser.add_argument('--hidden_layers', type=int, default=1024, help="number of hidden layers, defaults to 1024")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs, defaults to 10")
    parser.add_argument('--gpu', type=bool, default=True, help="indicates if the model should run on GPU")
    parser.add_argument('--save_directory',type=str, default="ModelCheckpoint", help="the folder where the checkpoint will be saved to, default to ModelCheckpoint")
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

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    if(args.gpu):
        # Find the device available to use using torch library
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create the model
    model, criterion, optimizer, scheduler = create_model(args.hidden_layers, args.arch, args.learning_rate)

    # Move model to the device specified above
    model.to(device)

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, args.epochs)
    
    # Save checkpoint
    save_checkpoint(model, model.classifier, scheduler, optimizer, args.epochs, image_datasets,  args.save_directory)

def create_model(hidden_layers=1024, arch="densenet161", learning_rate=0.001):
   
    input_size = 0
    optimizer = None
    # Load a pre-trained densenet network
    if(arch == "densenet161"):
        model = models.densenet161(pretrained=True)
        input_size = 2208
    elif (arch == "resnet18"):
        model = models.resnet18(pretrained=True)
        input_size = 512
    else:      
        return

    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    if arch == 'densenet161':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    elif arch == 'resnet18':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
   
    criterion = nn.NLLLoss()  
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return model, criterion, optimizer, scheduler


# train_model function based on :  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.htm - accessed 03-Jun-2020
# All sample code has been thoroughly understood.
# Train the classifier layers using backpropagation using the pre-trained network to get the features.
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, epochs=10):
     
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:

            print(phase)

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_checkpoint(model, classifier, scheduler, optimizer, epochs, image_datasets, save_dir="ModelCheckpoint"):
    
    # Save the checkpoint 
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'input_size': 2208,            
                  'output_size': 102,
                  'epochs': epochs,
                  'batch_size': 64,
                  'model': models.densenet161(pretrained=True),
                  'classifier': classifier,
                  'scheduler': scheduler,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
    
    torch.save(checkpoint, save_dir + "/" + "checkpoint.pth")

if __name__ == "__main__":
    main()
