# SL2NL - Sign Language to Natural Language 
This project has been developed in partial fulfilment of the requirements for the BEng (Hons) Software Engineering degree at the University of Westminster, all rights belong to the University. 

This project aims to investigate Gesture Recognition techniques and approaches in order to classify Sign Language gestures and interpret them in Natural Language text.

The implementation initially aimed to use a Convolutional Neural Network and TensorFlow for Machine Learning; however, a trade-off between accuracy and complexity was made and the implementation was achieved using the K – Nearest Neighbours algorithm. 

The proof of concept is a functioning prototype with basic usability that can be successfully trained on small amounts of data providing a strong foundation from which to further build upon.

# Updates 
The project has been represented at Westminster International University in Tashkent, Uzbekistan. And it has been featured on the local news, it can be found [here](https://www.youtube.com/watch?v=negbylETJCM).

## Requirements
Python Version 3.7.2

```python
import cv2
import os
import mahotas
import imutils
import numpy as np
import pandas as pd
import time
import threading
import datetime
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
```

## Basic User Guide

A video guide: https://www.youtube.com/watch?v=1b-5YD1-O40&feature=youtu.be

```
1. Ensure that the data folder, “train_images”, is empty. 
2. On launching the application, the first step is to provide training examples. 
      - Press ‘N’ - add new gesture name
      - Press ‘S’ - to start capturing the gesture (100 frames per gesture) 
      - Ensure the following gestures are included:
            -- ‘thumbsUp’ – to start prediction
            -- ‘stop’ – to stop prediction
            -- ‘other’ – catch-all category (idle state, arms by the side of the body)
      - Ensure there are enough gestures to create a sentence.
3. Once data has been collected, training is ready to begin; 
   (this can take a couple of minutes depending on how many gestures are to be trained.) 
      - Press ‘T’ – start training the model
4. Once training is completed, the prediction mode is available. 
      - Press ‘P’ – start prediction mode
      - ‘thumbsUp’ – will start prediction for a sentence
      - Once the accuracy threshold is crossed, the other gestures will be displayed in the terminal window. 
            (e.g. ‘hello’, ‘whats, ‘the time’)
      - ‘stop’ – stops the current prediction and will print out all the gestures done in order since the prediction was started 
            (e.g. ‘hello whats the time’)
5. Press ‘H’ to show a menu in the terminal window.
6. Press ‘Esc’ to stop the application
```

# Future Plans
I will continue working on this project in my spare time

## Short Term 
- Implement feature extraction with an already trained CNN, e.g. Inception V3 / MobileNet
- Allow models to be saved and loaded 
- Create an user front end with Django / Flask

## Long Term 
- Use services such as AmazonMechanicalTurk in order to gather a bigger dataset for Sign Lanugages
- Implement a Convolutional Neural Network from scratach to take full advantage
- Create a mobile application 
