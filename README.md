# CAPTCHA Image Prediction Project

## Overview
This project implements a Convolutional Recurrent Neural Network (CRNN) to predict the text content of CAPTCHA images. The model combines convolutional layers, recurrent layers, and a final dense layer to process and predict sequences of characters from image data.

## Dataset
The dataset consists of images with alpha-numeric characters used in CAPTCHA. Each image is labeled with the text it contains. The dataset is split into training (80%), validation (10%), and testing (10%) sets.

#### Link of the Dataset: [click](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images?rvi=1)

## Dependencies
- Python 3.10.6
- PyTorch (torch and torchvision)
- Pillow (PIL)
- NumPy
- pandas
- Matplotlib
- scikit-learn
- torchsummary

## Model Architecture
The model utilizes a pretrained ResNet-18 model for the convolutional part, followed by two GRU layers for the recurrent part, and a linear layer for character classification.

## Running the Code
The main script processes the images, creates the model, and runs the training and validation cycles. Here are the steps to execute the project:

1. Set the data directory where the dataset images are stored.
2. Load the dataset and split it into train, validation, and test sets.
3. Initialize the device to run the model on GPU if available.
4. Define the CRNN model with the appropriate number of characters and hidden layer size.
5. Initialize the model weights and move the model to the selected device.
6. Define the loss function and optimizer.
7. Run the training and validation process, logging the losses for each epoch.
8. Visualize the training and validation losses.
9. Evaluate the model's performance on the training, validation, and test sets.
10. Apply post-processing to correct predictions and evaluate the final accuracy.

## Usage
To run this project, ensure that the data path is set to the location of your CAPTCHA dataset, and then execute the script. Make sure all the dependencies are installed in your environment.

## Acknowledgements
This project was created using the ResNet architecture from PyTorch's torchvision module and adapted for CAPTCHA prediction.
