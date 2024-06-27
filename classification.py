#Add CSV creating script that will record image names and it's label. Also can make a copy model. Image to vec
#https://alirezasamar.com/blog/2023/03/fine-tuning-pre-trained-resnet-18-model-image-classification-pytorch/
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold as K_Fold  #RepeatedKFold
import torch
import torchvision.models as models 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

####Modifiable variables#######
batch_size = 64
num_epochs = 5
num_classes = 13
learning_rate = 0.01
num_of_splits = 3           #Number of splits in the cross validation
number_of_repeats = None    #If using RepeatedKFold, use this to set the number of repeats
no_weights_exist = False    #Set to True if you want to recreate the model everytime
labels_map = {
    0: "Aggregate",
    1: "Blurry",
    2: "Camera_Ring",
    3: "Ciliate",
    4: "Copepod",
    5: "Diatom:_Long_Chain",
    6: "Diatom:_Long_Single",
    7: "Diatom:_Spike_Chain",
    8: "Diatom:_Sprial_Chain",
    9: "Diatom:_Square_Single",
    10: "Dinoflagellate:_Circles",
    11: "Dinoflagellate:_Horns",
    12: "Phaeocystis"
}


## Create parse argues ##
parser = argparse.ArgumentParser(description = 'Select between training and categorizing')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', action='store_true', help='Train the model on the current training data.')
group.add_argument('-c', '--categorize', action='store_true', help='Categorize unknown data to make new training data')
args = parser.parse_args()

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def create_image_csv(dataset):
    df = pd.DataFrame

def train(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, num_epochs):
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

        # Iterate over the batches of the validation loader
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

        


#Load the resnet18 model on first run unless a pre-run model is found
if (not os.path.exists('HM_model.pth')):
    print("Previous model does not exist, loading resnet18")
    model = models.resnet18(weights='DEFAULT')#, weights='HM_weights')
    no_weights_exist = True
else:
    print("Previous model found, loading HM_model")
    model  = torch.load('HM_model.pth')



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
    kf = K_Fold(n_splits = num_of_splits) #n_repeats = 5,

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

    if no_weights_exist:
        torch.save(model, 'HM_model.pth')

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
        
        # Create folder if it doesn't exist
        label_folder = os.path.join('Categorized_Data', predicted_label)
        os.makedirs(label_folder, exist_ok=True)
        
        # Save the image to the corresponding folder
        destination_path = os.path.join(label_folder, img_name)
        os.rename(img_path, destination_path)
        print(f'Saved {img_name} to {label_folder}')

else:
    print("Exploring dataset")

    #Load the dataset
    dataset = ImageFolder(root = 'Training_Data')
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    


    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset
    # Test to see how to access dataset matterial
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()




