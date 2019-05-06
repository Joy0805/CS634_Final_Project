# CS634_Final_Project
The project is about Kaggle competition - RSNA Pneumonia Detection Challenge provided in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. 
For better understanding in the tutorial, please combine tutorial with comments in the notebook. 

## Project Background
- Pneumonia is one of the top 10 causes of death in the United States. According to the definition in Wikipedia,  pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs and the diagnosis is often based on the symptoms and physical examination. In this project, we built an algorithm to diagnose pneumonia through locating the lung opacity on chest radiographs(CXR). 

## Introduction
- In this project, we first chose convolutional neural network to segment the image of lung. The network consists of a number of residual blocks with convolutions and downsampling blocks with max pooling. At the end of the network, a single upsampling layer converts the output to the same shape as the input. Then we used connected components to separate multiple areas of predicted pneumonia, and drew a bounding box around every connected component.


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
  - All files available in https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data. 

## Step 1 - Collect information and separate training set
- Read the label information from stage_2_train_labels.csv
  - Create a dictionary pneumonia_location to store the information. 
- Retrieve information by sample name, location for all the samples with pneumonia
  - For those dicoms that detected pneumonia, add each location information to each patient id. Note that there might be multiple locations in a single dicom file(or say for each patient).
- Read and shuffle the original training set and divide it into new training set and validation set
  - Division criterion: the size of validation set was 1000. 

## Step 2 - Visualization of the overall information about chest radiographs
- Count the number of pneumonia areas in each pneumonia sample
  - Most patients had 2 pneumonia areas. 
- Draw heatmap of all pneumonia areas
  - X was the upper-left horizontal coordinate of the bounded area, y was the upper-left vertical coordinate of the bounded area, and w, h refer to width and height of the opacity. So the areas were bounded in horizontal coordinate x to x+w and vertical coordinate y to y+h. The brighter color in the heatmap meant the more overlapped area in the dicom. 
- Draw histogram about height and width of all pneumonia areas
  - The two graphs showed the distribution of height and width of the pneumonia area. 

## Step 3 - Create class generator
- Define basic information of each image file
- Define how to load dicom file as numpy array by using package pydicom and relate it with location information if the image contains pneumonia
- Define prediction function 
- Define how to get the prediction result
- Define epoch in order to learn and precit step by step
- Define the number of images need to learn and predict in each epoch

## Step 4 - Define convolution neural networks
- Define the normalization function of the neural networks(keras.layers.BatchNormalization)
  - Keras.layers.batchnormalization: Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
  - axis: Integer, the axis that should be normalized (typically the features axis). After a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization. Because we used conv2D, the axis we used is 1
  - momentum: Momentum for the moving mean and the moving variance.
  - epsilon: Small float added to variance to avoid dividing by zero.

- Define The hidden layer of the activation function of the neural networks(keras.layers.LeakyReLU)
  - Keras.layers.leakyRelLU: Leaky version of a Rectified Linear Unit. It allows a small gradient when the unit is not active: f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
  - alpha: float >= 0. Negative slope coefficient.

- Define The convolution layer of the neural networks(keras.layers.Conv2D)
  - Keras.layers.Conv2D： This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well. When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
  - filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
The filters we used is “channles”, which is defined at next step named “create network and compiler”.
  - kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. It can be a single integer to specify the same value for all spatial dimensions.
  - padding: one of "valid" or "same" (case-insensitive). Note that "same" is slightly inconsistent across backends with strides != 1. 
  - use_bias: Boolean, whether the layer uses a bias vector.

- Define the normalization and max pooling function of neural networks to reduce feature dimensions and avoid over-fitting(keras.layers.MaxPool2D)
  - Keras.layers.MaxPool2D: Max pooling operation for spatial data.
  - pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.

- Define the channels, blocks and depth of the neural networks
  - The reason we choose channels = 16, n_blocks = 4, depth = 3 is that under the same complexity, the loss function of the output result using these parameters is relatively small, indicating that the data fitting degree is relatively high, and the data running speed is relatively fast. Even if the complexity is increased, the loss function does not change significantly. Therefore, the selection of this set of parameters is reasonable.

## Step 5 - Learning by applying convolution neural networks
- Define the jaccard loss function
- Define the iou function
- Create network and compiler by using channels=16, n_blocks=4 and depth=3
- Define cosine learning rate
- Create train and validation generators and record the validation results

## Step 6 - Show prediction performance epoch by epoch
- Show the value of jaccard loss function
  - As the training process going by, the value of both train and valid loss function was decreasing.
- Show the prediction accuracy
  - As the training process going by, the accuracy was continuously rising roughly with the peak accuracy as 0.9723. 
  - The valid accuracy and the train accyracy are constantly approaching.
- Show the overlapping area rate of predicted area devided by actual area(iou)
  - As the training process going by, the mean value of iou was ascending roughly with the peak accuracy as 0.7419.
* All these results showed that our convolution neural network has a satisfectory performance.

## Step 7 - Show prediction performance in cases
- Show the predicted bounding boxes in several samples of a batch and compare it with the actual bounding boxes
  - The blue bounding boxes presect the actual pneumonia areas, and the red bounding boxes presect the predicted pneumonia areas.
  - According to the presented images, it is clearly that the sensitivity is pretty well. All the actual pneumonia areas were predicted correctly.
  - However, the specificity is not good as the sensitivity, several non-pneumonia areas were regarded as pneumonia areas incorrectly.

## Step 8 - Prediction
- Forecasting test sets with trained models
  - A generator for test set was initialized. A dictionary named submission_dict was created to stored the predictions.  A nested for loop was used to append multiple pneumonia areas detected in the dicoms. The loop was exited when it collected all test results. 

## Performance Evaluation
- The submitted result was graded by Kaggle, and here are our scores. 
  - Private Score: 0.09519
  - Public Score: 0.06712

## Reference
- CNN Segmentation + connected components, Jonne, https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components. 
