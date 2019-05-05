# CS634_Final_Project
The project is about Kaggle competition - RSNA Pneumonia Detection Challenge provided in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. 

## Project Background
Pneumonia is one of the top 10 causes of death in the United States. According to the definition in Wikipedia,  pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs and the diagnosis of pneumonia is often based on the symptoms and physical examination. In this project, we built an algorithm to diagnose pneumonia based on chest radiographs(CXR). The objective of the algorithm is to locate the lung opacity on chest radiographs and diagnose pneumonia.

## Introduction
- The objective of the analysis is to locate the lung opacity on chest radiographs and diagnose pneumonia. 
- Firstly, choosing convolutional neural network to segment the image of lung.  The network consists of a number of residual blocks with convolutions and downsampling blocks with max pooling.  At the end of the network a single upsampling layer converts the output to the same shape as the input.
- Secondly, using connected components to deparate multiple areas of predicted pneumonia.
- Finally, drawing a bounding box around every connected component.


## Overview
- Load pneumonia locations
- Analyze chest radiographs
- Create methods for data training
- Predict on validation
- Predict on test set

## Required Packages
- os
- csv
- random
- pydicom
- numpy
- pandas
- skimage
- tensorflow
- matplotlib

## Files and Folders Explanation
- stage_2_train_labels.csv: training data set
- stage_2_train_images: dicom files of training set
- stage_2_test_images: dicom files of test set
* All files available in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data. 

## Step 1 - Collect information and separate training set
- Read the label information from stage_2_train_labels.csv
- Retrieve information by sample name, location, and whether there is pneumonia
- Pick out all the samples with pneumonia
- Read and shuffle the original training set and divide it into new training set and validation set
-- The division criterion of validation set is that it is one tenth of training set

## Step 2 - Visualization of the overall information about chest radiographs
- Count the number of pneumonia areas in each pneumonia sample
- Draw heatmap of all pneumonia areas
- Draw histgram of height and width of all pneumonia areas

## Step 3 - Create class generator
- Define basic information of each image file
- Define how to load dicom file as numpy array by using package pydicom and relate it with location information if the image contains pneumonia
- Define prediction function 
- Define how to get the prediction result
- Define epoch in order to learn and precit step by step
- Define the number of images need to learn and predict in each epoch

## Step 4 - Define convolution neural networks
- Define the normalization function of the neural networks(keras.layers.BatchNormalization)
- Define The hidden layer of the activation function of the neural networks(keras.layers.LeakyReLU)
- Define The convolution layer of the neural networks(keras.layers.Conv2D)
- Define the normalization and max pooling function of neural networks to reduce feature dimensions and avoid over-fitting(keras.layers.MaxPool2D)
- Define the channels, blocks and depth of the neural networks

## Step 5 - Learning by applying convolution neural networks
- Define the jaccard loss function
- Define the iou function
- Create network and compiler by using channels=4, n_blocks=2 and depth=2
- Define cosine learning rate
- Create train and validation generators and record the validation results

## Step 6 - Show prediction performance epoch by epoch
- Calculate the value of jaccard loss function
- Calculate the prediction accuracy
- Calculate the overlapping area between predicted area and actual area(iou)

## Step 7 - Predict
- Forecasting test sets with trained models


## Reference
- CNN Segmentation + connected components, Jonne, https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components. 
