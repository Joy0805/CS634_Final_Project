# CS634_Final_Project
The project is about Kaggle competition - RSNA Pneumonia Detection Challenge provided in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. 

## Introduction
- 
- The objective of the analysis is to locate the lung opacity on chest radiographs and diagnose pneumonia. 

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



## Step 1 - Collect information about pneumonia locations from csv file


## Step 2 - Show information about chest radiographs


## Step 3 - Define Class and methods


## Step 4 - Create train and validation generators


## Step 5 - Plot and Show Prediction on validation


## Step 6 - Predict





## Reference
- CNN Segmentation + connected components, Jonne, https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components. 
