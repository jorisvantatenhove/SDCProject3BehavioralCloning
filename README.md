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


[//]: #  (Image References)

[image1]: ./examples/centerlanedriving.png "Center Lane Driving"
[image2]: ./examples/recoverydriving.png "Recovery Driving"

##  Rubric Points
### We will describe all the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually, detailing the approach taken to tackle the problems.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project contains the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the provided drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### Model architecture

The final model architecture (model.py lines 53-64) is loosely inspired by the Nvidia network. This model has been proven to work well for similar problems, so using this design as the basis for this network seemed like a natural choice.

We start of with some convolution layers with a RELU activation function, followed by a bunch of fully connected layers (model.py lines 53-64). It includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). We also crop out irrelevant parts of the input. In order to prevent overfitting, we've added two dropout layers after the two biggest fully connected layers with a 50% probability of dropping a node. The final result is a single output node representing the steering angle.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Normalized Input      | 75x320 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, 24 depth 						 	|
| RELU					|												| 
| Convolution 5x5     	| 2x2 stride, 36 depth 						 	|
| RELU					|												| 
| Convolution 3x3     	| 2x2 stride, 48 depth 						 	|
| RELU					|												| 
| Convolution 3x3     	| 2x2 stride, 64 depth 						 	|
| RELU					|												| 
| Convolution 3x3     	| 2x2 stride, 96 depth 						 	|
| RELU					|												|
| Flatten				|												|
| Fully connected		| outputs 1200									|
| Dropout				| Keep probability 0.5							|
| Fully connected		| outputs 600									|
| Dropout				| Keep probability 0.5							|
| Fully connected		| outputs 250									|
| Fully connected		| outputs 50s									|
| Fully connected		| outputs 1										|

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Earlier versions of the model did not include the dropout, but there were some serious overfitting issues, shown by the difference in error comparing the training to the validation set. 

The learning rate was not tuned manually, as the model uses an Adam optimizer.

#### Training data

Training data was chosen to keep the vehicle driving on the road. The training data contains three laps of driving around the track cleanly (first picture), one lap driving in the opposite direction and a lap where we just recorded recovery moves (second picture, notice the steep angle to the left as we are on the right side of the road).

![Center Lane Driving][image1]

![Recovery Driving][image2]

After teaching and tuning the model, we discovered we had some issues with the sharp corners. To combat this, we added some extra corner data where we take corners relatively steeply.

Moreover, we record three images when driving: the center image, used as input when actually driving autonomously, and a left and a right image. We add these images with a modified angle: we add or substract a correction of 0.3. By including this, the model teaches the car to steer right when it's on the left of the track and also vice versa.

We played around with flipping the data set and negativing the steering angle, but this did not really improve the performance more than adding a new lap did, so we did not think it's worth adding this compared to the extra calculation time you get - it's more efficient to just add more 'actual' data.

Track 2, a more challenging track as it includes shadows, a middle line edge, hills and lanes right next to each other, was completely ignored when getting input data for Track 1.

The input data is shuffled before we start training, and use 20% of the input data for validation. As mentioned, we do not reserve any data for testing, but instead test on the actual track itself. We use a single epoch, as that seemed enough to train this model.

