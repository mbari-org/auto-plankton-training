# Machine Learning creation code

The [mbari-auto-plankton-training](https://github.com/mbari-org/auto-plankton-training) package allows to both train a new model and to classify new images for bootstrapping a dataset.

To install the required packages, set up a python environment and then pip into a terminal:
'''shell
pip install requirements.txt
'''
### Training
Training Data needs to be in a file called "Training Data"
'''shell
mkdir "Training Data"
'''

In order to train a model, match the training categories in classification.py with the categories in the dir "Training Data". Once this is done and training data exists, type into the terminal that is in the directory with both "Training Data" and classification.py
'''shell
classification.py -t
'''
The -t flag will set the programing into training mode to create a new model if no previous model exists. It will both look and save a model called HM_model.pth. If a model does not exist, then classification.py will create it.

### Classification
In order to with HM_model.pth to classify new data for bootstrapping, images must be in a file called "New Data"
'''shell
mkdir "New Data
'''
Dump all images that are to be categorized into this directory. Do not place other directories into this directory as classification.py will not see those images. 

To start to classification process, type into the terminal that is in the directory with both "New Data" and classification.py
'''shell
classification.py -c
'''
The -c flag will set the programing into classification mode. A model must exist called HM_model.pth and be in the same directory as classification.py. All classifed images will be placed into a new directory called "Categorized Data". The program will make this directory and does not need to exist before running classification.py. All images will be placed into directories with their associated categorized name as specified in classification.py.

### History
Every completed run of classification.py will result in a History directory to be created. This directory will included previous models created, CSVs of images that the models were trained on, and CSVs of images that where categorized depending on the command line arguments given to classification.py