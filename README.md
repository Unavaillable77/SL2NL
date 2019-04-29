# SL2NL
Basic User Guide
A video guide: YouTube video

•	Before launching the project, ensure that the training folder is empty. 
•	The first step is to provide training examples. Using the terminal controls, add gesture names and use the camera window to capture yourself doing that gesture until 100 frames would be captured. (This does not take too long, ensure you have enough gestures to create a sentence.)
•	A start and stop gesture has to be trained, in this case, “thumbsUp” is used to start the prediction and “stop” is used to stop it. 
•	Before finishing ensure that you also have a gesture called “other” which will be just you in frame doing no gesture at all with the hands by your side. 
•	Once data has been collected, training is ready to begin, this can take couple minutes depending on how many gestures you intend to train. 
•	Once training is completed, you can enter the prediction mode. It will capture frames from the camera feed and use the predict function every half a second to try get a prediction.
•	If a certain prediction threshold is crossed, it will append the gesture label to the prediction list. 
•	Using the start gesture you can start the prediction, after you’ve completed the sentence you can stop using the stop gesture. This will print out the gestures you’ve completed in the order you’ve completed them in 
