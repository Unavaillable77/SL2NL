# SL2NL
Basic User Guide
A video guide: https://www.youtube.com/watch?v=1b-5YD1-O40&feature=youtu.be

1.	Ensure that the data folder, “train_images”, is empty. 
2.	On launching the application, the first step is to provide training examples. 
      o	Press ‘N’ - add new gesture name
      o	Press ‘S’ - to start capturing the gesture (100 frames per gesture) 
      o	Ensure the following gestures are included:
          	‘thumbsUp’ – to start prediction
          	‘stop’ – to stop prediction
          	‘other’ – catch-all category (idle state, arms by the side of the body)
      o	Ensure there are enough gestures to create a sentence.
3.	Once data has been collected, training is ready to begin; this can take a couple of minutes depending on how many gestures are to be       trained. 
      o	Press ‘T’ – start training the model
4.	Once training is completed, the prediction mode is available. 
      o	Press ‘P’ – start prediction mode
      o	‘thumbsUp’ – will start prediction for a sentence
      o	Once the accuracy threshold is crossed, the other gestures will be displayed in the terminal window. (e.g. ‘hello’, ‘whats, ‘the           time’)
      o	‘stop’ – stops the current prediction and will print out all the gestures done in order since the prediction was started (e.g.             ‘hello whats the time’)
5.	Press ‘H’ to show a menu in the terminal window
6.	Press ‘Esc’ to stop the application
