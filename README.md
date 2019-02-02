# Facial_Expression_Detection
To identify the facial expressions given at runtime using web-cam

## ABOUT THE DATASET:
Dataset(FER 2013) Link :https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge.

Dataset consists of 48x48 pixel grayscale images of faces.

In this repository, I used Convolution Neural Network to get the facial expression of faces present in the image.

The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

## ABOUT THE CODE:
  1. Getting the dataset and splitting the dataset into training and test data.
  2. Scaling the data.
  3. Buliding a Convolution Neural Network and fitting the model to training data.
  4. Using OpenCV and Haar Cascade to identify the faces present in the image.
  5. Passing the cropped faces into the network and predicting the expression.

Accuracy reached : 63%
