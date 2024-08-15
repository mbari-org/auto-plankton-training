#Author: Steven Patrick
#Add CSV creating script that will record image names and it's label. Also can make a copy model. Image to vec
#https://alirezasamar.com/blog/2023/03/fine-tuning-pre-trained-resnet-18-model-image-classification-pytorch/
import os

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold as K_Fold  #RepeatedKFold
import torch
import torchvision.models as models 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as pltz

from EarlyStopping import EarlyStopping

####Modifiable variables#######

batch_size = 64
num_epochs = 1000
learning_rate = 0.01
num_of_splits = 5           #Number of splits in the cross validation
num_of_repeats = 1    #If using RepeatedKFold, use this to set the number of repeats
remake_model = False    #Set to True if you want to recreate the model everytime
patience = 10            #Early Stopping patience
min_delta = 0.0       #The amount of change require to not early stop.
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
label_count = {
    "Aggregate": 0,
    "Bad_Mask": 0,
    "Blurry": 0,
    "Camera_Ring": 0,
    "Ciliate": 0,
    "Copepod": 0,
    "Diatom_Long_Chain": 0,
    "Diatom_Long_Single": 0,
    "Diatom_Spike_Chain": 0,
    "Diatom_Sprial_Chain": 0,
    "Diatom_Square_Single": 0,
    "Dinoflagellate_Circles": 0,
    "Dinoflagellate_Horns": 0,
    "Phaeocystis": 0,
    "Radiolaria": 0
}
num_classes = len(labels_map)
#model = None

## Create parse argues ##
parser = argparse.ArgumentParser(description = 'Select between training and categorizing')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', action='store_true', help='Train the model on the current training data.')
group.add_argument('-c', '--categorize', action='store_true', help='Categorize unknown data to make new training data')
args = parser.parse_args()


def create_image_csv():
    print("Saving CSV")

    folder_path = 'Training_Data'
    data = {
        "image": [],
        "label": []
    }
    #loop through training_data to see all the training dirs, then loop through the training dirs to get all the images. Then save image file name and it's file label in a csv.
    for dirs in os.listdir(folder_path):
        working_dir = os.path.join(folder_path, dirs)
        for files in os.listdir(working_dir):
            data["image"].append(files)
            data["label"].append(dirs)

    df = pd.DataFrame(data)
    # This creates the csv name: ISO date format
    filename = time.strftime("%Y-%m-%dT%H:%M.csv", time.gmtime())
    directory_name=time.strftime("%Y-%m-%d", time.gmtime())

    # Try to save the CSV, If you can't, error out.
    try:
        path = os.path.join("History", "CSVs")
        path = os.path.join(path, directory_name)
        os.makedirs(path, exist_ok = True)
        df.to_csv(os.path.join(path, filename), header=False, index=False)
    except OSError as error:
        print(error)

def create_cat_count_csv():
	folder_path = 'Categorized_Data'
	data = {
		"data": {"image": [],
			"label": []
			},
		"label_count": {"label": labels_map.values(),
				"count": label_count.values()
		}
	}
	#loop through training_data to see all the training dirs, then loop through the training dirs to get all the images. Then save image file name and it's file label in a csv.
	for dirs in os.listdir(folder_path):
		working_dir = os.path.join(folder_path, dirs)
		for files in os.listdir(working_dir):
			data["data"]["image"].append(files)
			data["data"]["label"].append(dirs)

	df = pd.DataFrame(data)
	# This creates the csv name: ISO date format
	filename = time.strftime("%Y-%m-%dT%H:%M.csv", time.gmtime())
	directory_name=time.strftime("%Y-%m-%d", time.gmtime())
	try:
		path = os.path.join("History", "Categorize_CSVs")
		path = os.path.join(path, directory_name)
		os.makedirs(path, exist_ok = True)
		#df = pd.DataFrame.from_dict(label_count, orient="index")
		df = df.T
		df.to_csv(os.path.join(path, filename), header=False, index=False)
	except OSError as error:
		print(error)
    

def save_model(model):
    torch.save(model, 'HM_model.pth')
    # This creates the csv name: ISO date format
    filename = time.strftime("%Y-%m-%dT%H:%M_Model.pth", time.gmtime())
    directory_name=time.strftime("%Y-%m-%d", time.gmtime())

    # Try to save the CSV, If you can't, error out.
    try:
        path = os.path.join("History", "Models")
        path = os.path.join(os.path.join(path, directory_name))
        os.makedirs(path, exist_ok = True)
        torch.save(model, os.path.join(path, filename))
    except OSError as error:
        print(error)


def train(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # Create early stopping
    early_stopping = EarlyStopping(patience = patience, verbose = True, delta = min_delta)


    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0        

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate oHM_model_over_night.pthver the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print("The output was: " +  str(preds) + "\nAnd the label was: " + str(labels) + "\n")


        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

        

        if early_stopping(val_loss, model):
            print("\nEarly Stopping Here\n")
            break


    model = early_stopping.load_checkpoint(model).to(device)

        


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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #This changes the pixel to have mean of 0 and a std of 1. We will probably need to change this for our dataset. subtract the mean and divide by the std.
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
