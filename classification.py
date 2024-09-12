"""
Author: Steven Patrick
Description:
    This script handles two main functionalities: training a model using a pre-trained ResNet-18 and categorizing images into classes using a trained model. 
    Additionally, it allows creating CSV files to track image filenames and their corresponding labels. The script supports early stopping and cross-validation.
    
    The two main functionalities:
    1. Training:
        - Uses cross-validation and early stopping.
        - Trains the last fully connected layer of a pre-trained ResNet-18 model.
        - Optionally fine-tunes the entire network.
        - Tracks the training and validation accuracy and loss.
        - Saves a model checkpoint and the best-performing model.
        - Outputs the results in a CSV format for later inspection.
    2. Categorizing:
        - Classifies images in the 'New_Data' folder into categories.
        - Moves images to a folder based on the predicted class.
        - Keeps track of the count of images per category and exports it as a CSV.
        
    Additional Features:
        - Early stopping to avoid overfitting during training.
        - Option to reload a saved model or start from scratch.
        - CSV logging for the images and their labels, as well as a count of images in each category.
    
    To use the script:
        - Run with the argument `-t` or `--train` to train the model.
        - Run with the argument `-c` or `--categorize` to categorize images.
        - The script will automatically detect if a saved model exists and load it, unless you set the `remake_model` variable to True.
        - CSVs will be created in the "History" directory, organized by date.
    
Reference:
    Fine-tuning a pre-trained ResNet-18 model for image classification using PyTorch:
    https://alirezasamar.com/blog/2023/03/fine-tuning-pre-trained-resnet-18-model-image-classification-pytorch/

"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold as K_Fold
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt

from earlystopping import EarlyStopping

#### Modifiable variables ####

# Training parameters
batch_size = 64  # Batch size for DataLoader
num_epochs = 1000  # Maximum number of epochs to train
learning_rate = 0.01  # Learning rate for optimizer
num_of_splits = 5  # Number of splits in K-fold cross-validation
num_of_repeats = 1  # Number of times K-fold cross-validation is repeated
remake_model = False  # Set to True if a new model is to be trained every time
patience = 10  # Early stopping patience (number of epochs without improvement)
min_delta = 0.0  # Minimum delta for improvement to reset early stopping counter

# Label mapping (class labels)
labels_map = {
    0: "Aggregate",
    1: "Bad_Mask",
    2: "Blurry",
    3: "Camera_Ring",
    4: "Ciliate",
    5: "Copepod",
    6: "Diatom_Long_Chain",
    7: "Diatom_Long_Single",
    8: "Diatom_Spike_Chain",
    9: "Diatom_Sprial_Chain",
    10: "Diatom_Square_Single",
    11: "Dinoflagellate_Circles",
    12: "Dinoflagellate_Horns",
    13: "Phaeocystis",
    14: "Radiolaria"
}

# Counter for labeled data
label_count = {label: 0 for label in labels_map.values()}

# Number of classes in the dataset
num_classes = len(labels_map)


def create_image_csv():
    """
    Create a CSV file that records image names and their corresponding labels from the 'Training_Data' directory.
    
    The CSV file is saved in the 'History/CSVs' directory, organized by the current date. The filename is based on the
    ISO date format.
    """
    print("Saving CSV")

    folder_path = 'Training_Data'
    data = {
        "image": [],
        "label": []
    }

    # Iterate through all subdirectories in 'Training_Data', adding image filenames and labels to data
    for dirs in os.listdir(folder_path):
        working_dir = os.path.join(folder_path, dirs)
        for files in os.listdir(working_dir):
            data["image"].append(files)
            data["label"].append(dirs)

    df = pd.DataFrame(data)

    # Save CSV with filename based on ISO date format
    filename = time.strftime("%Y-%m-%dT%H:%M.csv", time.gmtime())
    directory_name = time.strftime("%Y-%m-%d", time.gmtime())

    try:
        path = os.path.join("History", "CSVs", directory_name)
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, filename), header=False, index=False)
    except OSError as error:
        print(error)


def create_cat_count_csv():
    """
    Create a CSV file that records the counts of categorized images based on their predicted labels from the 
    'Categorized_Data' directory.
    
    The CSV file is saved in the 'History/Categorize_CSVs' directory, organized by the current date.
    """
    folder_path = 'Categorized_Data'
    data = {
        "data": {"image": [], "label": []},
        "label_count": {"label": labels_map.values(), "count": label_count.values()}
    }

    # Iterate through all subdirectories in 'Categorized_Data', adding image filenames and labels to data
    for dirs in os.listdir(folder_path):
        working_dir = os.path.join(folder_path, dirs)
        for files in os.listdir(working_dir):
            data["data"]["image"].append(files)
            data["data"]["label"].append(dirs)

    df = pd.DataFrame(data)

    # Save CSV with filename based on ISO date format
    filename = time.strftime("%Y-%m-%dT%H:%M.csv", time.gmtime())
    directory_name = time.strftime("%Y-%m-%d", time.gmtime())

    try:
        path = os.path.join("History", "Categorize_CSVs", directory_name)
        os.makedirs(path, exist_ok=True)
        df = df.T
        df.to_csv(os.path.join(path, filename), header=False, index=False)
    except OSError as error:
        print(error)


def save_model(model):
    """
    Save the trained model as a .pth file. The model is saved in the 'History/Models' directory, organized by the 
    current date. The filename is based on the ISO date format.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model to be saved.
    """
    torch.save(model, 'HM_model.pth')
    filename = time.strftime("%Y-%m-%dT%H:%M_Model.pth", time.gmtime())
    directory_name = time.strftime("%Y-%m-%d", time.gmtime())

    try:
        path = os.path.join("History", "Models", directory_name)
        os.makedirs(path, exist_ok=True)
        torch.save(model, os.path.join(path, filename))
    except OSError as error:
        print(error)



def train(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, num_epochs):
    """
    Train and validate the given model using the provided training and validation data loaders.
    This function implements a standard training loop in PyTorch, along with early stopping to 
    prevent overfitting.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        train_dataset (Dataset): Full training dataset.
        val_dataset (Dataset): Full validation dataset.
        criterion (torch.nn.Module): Loss function used to calculate the training and validation losses.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model's parameters.
        num_epochs (int): The number of epochs for which to train the model.

    Functionality:
    - Performs a forward pass through the network for each batch of training data.
    - Computes the loss and performs a backward pass to update the model's parameters.
    - Evaluates the model on the validation dataset at the end of each epoch.
    - Implements early stopping based on validation loss, to prevent overfitting.
    - Loads the best-performing model based on validation performance after early stopping.

    Training Process:
    1. Set the model to training mode.
    2. For each epoch, iterate over the training batches:
       - Compute predictions, loss, and perform backpropagation.
       - Update the model parameters.
    3. At the end of each epoch, evaluate the model on the validation dataset:
       - Set the model to evaluation mode.
       - Compute predictions and loss on the validation data.
    4. Print epoch statistics such as training loss, validation loss, and accuracy.
    5. If early stopping is triggered, stop training and load the best-performing model.

    Returns:
        torch.nn.Module: The trained model, potentially with early stopping applied.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create early stopping object to monitor validation loss
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=min_delta)

    # Training loop over epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        running_loss = 0.0  # Tracks loss across batches
        running_corrects = 0  # Tracks correct predictions across batches

        # Iterate through the training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

            optimizer.zero_grad()  # Reset gradients from the previous step

            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get the predictions
            loss = criterion(outputs, labels)  # Compute the loss

            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            running_corrects += torch.sum(preds == labels.data)  # Accumulate correct predictions

        # Compute average training loss and accuracy for the epoch
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Validation phase: set the model to evaluation mode
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0

        # Disable gradient calculation for validation (saves memory and computation)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)  # Compute validation loss

                running_val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
                running_val_corrects += torch.sum(preds == labels.data)  # Accumulate validation accuracy

        # Compute average validation loss and accuracy
        val_loss = running_val_loss / len(val_dataset)
        val_acc = running_val_corrects.double() / len(val_dataset)

        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
              f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        # Check if early stopping is triggered based on validation loss
        if early_stopping(val_loss, model):
            print("\nEarly stopping triggered, halting training.\n")
            break

    # Load the best model after early stopping
    model = early_stopping.load_checkpoint(model).to(device)
    return model

## Create parse argues ##
parser = argparse.ArgumentParser(description = 'Select between training and categorizing')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', action='store_true', help='Train the model on the current training data.')
group.add_argument('-c', '--categorize', action='store_true', help='Categorize unknown data to make new training data')
args = parser.parse_args()


#Load the resnet18 model on first run unless a pre-run model is found
if (not os.path.exists('HM_model.pth') or remake_model):
    print("Previous model does not exist, loading resnet18")
    model = models.resnet18(weights='DEFAULT')#, weights='HM_weights')
    #Modify the last layer of the model
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
else:
    print("Previous model found, loading HM_model")
    model = models.resnet18(weights='DEFAULT')#, weights='HM_weights')
    #Modify the last layer of the model
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('HM_model.pth'))
    #print(model)



# Define the transformations to apply to the images
transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #This changes the pixel to have mean of 0 and a std of 1. We will probably need to change this for our dataset. subtract the mean and divide by the std.
    transforms.RandomHorizontalFlip([0.5]),
    transforms.RandomVerticalFlip([0.5]),
    transforms.RandomChannelPermutation(),
    transforms.ColorJitter(),
    transforms.GaussianNoise()
])

# Set the device
print("Cuda Found: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

###### Training #######
if args.train:

    #set up cross-validation
    kf = K_Fold(n_splits = num_of_splits, n_repeats = num_of_repeats) #n_repeats = 5,

    #Freeze all the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    #Modify the last layer of the model
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    #Load the dataset
    dataset = ImageFolder(root = 'Training_Data', transform=transform)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)


    #Cross_Validation
    for fold, (train_index,val_index) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')

        #Create the train and the evaluate subsets from the cross-validation split
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        train(model, train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs=num_epochs)

    # Unfreeze all the layers and fine-tune the entire network for a few more epochs
    for param in model.parameters():
        param.requires_grad = True

    # Fine-tune the last layer for a few epochs
    for fold, (train_index,val_index) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')

        #Create the train and the evaluate subsets from the cross-validation split
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        train(model, train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs=num_epochs)

    
    create_image_csv()
    save_model(model)

###### Categorizing ######
elif args.categorize:
    # Load the dataset from 'New_Data' folder
    folder_path = 'New_Data'

    # Set the model to evaluate mode and disable the tensor.backward() call
    model.eval()  # Set the model to evaluation mode
    torch.no_grad()
    # Iterate over each image in the folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = Image.open(img_path)
        image_tensor = transform(image).unsqueeze(0).to(device)  # Apply transformation and move to device
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

        ## Using the label map defined at the top for the folder names for the labels.
        predicted_label = labels_map[predicted_label]
        label_count[predicted_label] += 1 
        
        # Create folder if it doesn't exist
        label_folder = os.path.join('Categorized_Data', predicted_label)
        os.makedirs(label_folder, exist_ok=True)
        
        # Save the image to the corresponding folder
        destination_path = os.path.join(label_folder, img_name)
        os.rename(img_path, destination_path)
        print(f'Saved {img_name} to {label_folder}')
    print(label_count)
    create_cat_count_csv()

else:
	print(labels_map.values())
	print("error: select train or caregorize.")
