# CS634_Final_Project
The project is about Kaggle competition - RSNA Pneumonia Detection Challenge provided in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. 

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

## Important Variables Explanation



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

## Step 3 - Define Class and methods


## Step 4 - Create train and validation generators


## Step 5 - Show prediction performance epoch by epoch
- Calculate the value of jaccard loss function
- Calculate the prediction accuracy
- Calculate the overlapping area between predicted area and actual area(iou)

## Step 6 - Predict





## Reference
- CNN Segmentation + connected components, Jonne, https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components. 
