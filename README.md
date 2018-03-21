# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
    * model.h5.\* show the process of improvement
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of
* Convolutional layer with 5x5 and depth 24 (clone.py lines 55-59) and a dropout (ratio 0.8) and maxpooling 2x2
* Convolutional layer with 5x5 and depth 36 (clone.py lines 61-65) and a dropout (ratio 0.7) and maxpooling 2x2
* Convolutional layer with 5x5 and depth 48 (clone.py lines 67-71) and a dropout (ratio 0.7) and maxpooling 2x2
* Convolutional layer with 3x3 and depth 64 (clone.py lines 67-71) and a dropout (ratio 0.7)
    * It adopts 'same' instead of 'valid' to create one more later convolutional layer
* Convolutional layer with 3x3 and depth 64 (clone.py lines 67-71) and a dropout (ratio 0.7)
* Full-connected layer with 100 (clone.py lines 87-89) neurons and a dropout (ratio 0.5)
* Full-connected layer with 50 (clone.py lines 92-94) neurons and a dropout (ratio 0.5)
* Full-connected layer with 10 (clone.py lines 97-98) neurons and a dropout (ratio 0.5)

Each layer includes RELU layer to introduce nonlinearity except the last full-connected layer.
The data is normalized in the model using a Keras lambda layer (clone.py line 52).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 58, 64, 70, 76, 81, 89, 94, 98). 

The model was trained and validated on different data sets to ensure
that the model was not overfitting (code line 104, model.fit() with validation\_split=0.2).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
See run1.mp4 or [this youtube link](https://youtu.be/pYz9YYp5ubE).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 103).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to
* Gather data while driving on the road (using center, left, and right cameras).
* Train data based on the deep convolution neural network.
* Test the trained model by driving the car on the track.

To combat the overfitting, I added dropouts to each layer of the model.

When I first trained the model, the result algorithm makes the car out of the track.
It is because my data collection has several problems:
* When collecting the data, I drove the car with full speed and It resulted in a small amount of data.
* I used keyboard direction keys to turn left and right which resulted in not smooth direction changes.
* I only drove the car on the track without data to return to the track.

My strategy to improve the result were:
* To drive slowly when gathering the data.
* To make sure the data contains both the way to drive on the track and how to return back to the track if it is not in the center.

#### 2. Final Model Architecture

Convolution Neural Network with:
* Convolutional layer with 5x5 and depth 24 (clone.py lines 55-59) and a dropout (ratio 0.8) and maxpooling 2x2
* Convolutional layer with 5x5 and depth 36 (clone.py lines 61-65) and a dropout (ratio 0.7) and maxpooling 2x2
* Convolutional layer with 5x5 and depth 48 (clone.py lines 67-71) and a dropout (ratio 0.7) and maxpooling 2x2
* Convolutional layer with 3x3 and depth 64 (clone.py lines 67-71) and a dropout (ratio 0.7)
    * It adopts 'same' instead of 'valid' to create one more later convolutional layer
* Convolutional layer with 3x3 and depth 64 (clone.py lines 67-71) and a dropout (ratio 0.7)
* Full-connected layer with 100 (clone.py lines 87-89) neurons and a dropout (ratio 0.5)
* Full-connected layer with 50 (clone.py lines 92-94) neurons and a dropout (ratio 0.5)
* Full-connected layer with 10 (clone.py lines 97-98) neurons and a dropout (ratio 0.5)

#### 3. Creation of the Training Set & Training Process

My strategy to gather the data were:
* To drive slowly when gathering the data.
* To make sure the data contains both the way to drive on the track and how to return back to the track if it is not in the center.

I gathered 18719 images and also used their flipped images.
As a result, the vehicle could stay on the track.
See run1.mp4 or [this youtube link](https://youtu.be/pYz9YYp5ubE).
