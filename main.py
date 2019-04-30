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

# This parameter controls number of image samples to be taken PER gesture
NUMOFSAMPLES = 100

# no. of nearest neighbours for classification
NEIGHBOURS = 7

# no. of jobs for k-NN distance (-1 uses all available cores)
JOBS = 1

# fixed-sizes for image
IMAGE_SIZE = (320, 240)

# path to training data
TRAIN_PATH = "train_images"

# bins for histogram
BINS = 8

# train_test_split size
TEST_SIZE = 0.10

# global variables
fullPred = []
guessGesture = False
saveImage = False
predicting = False
globalModel = 0
counter = 0
gestureName = ''
pastPrediction = ''
now = datetime.datetime.now().microsecond / 1000
thenSeconds = datetime.datetime.now().second

banner = '''What would you like to do ?
    N - Enter a new gesture name
    S - Save images for current gesture (Include: 'thumbsUp', 'stop', 'other')
    T - Train the model 
    P - Start the prediction mode (make sure model was trained)
    H - Show menu
    'Esc' - Exit	
         '''


# Creating the directory for each new gesture
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# feature-descriptor-1: Hu Moments (Shape Matching)
def moments(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(grey)).flatten()
    return feature


# feature-descriptor-2: Haralick (Texture extraction)
def haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    f_haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return f_haralick


# feature-descriptor-3: Color Histogram
def color_histogram(image):
    # extract a 3D color histogram from the HSV color space using the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()


# feature-descriptor-4: Raw Pixels
def feature_vector(image):
    # resize the image to a fixed size, now flatten the image into a list of raw pixel intensities
    return cv2.resize(image, IMAGE_SIZE).flatten()


def predict(frame):
    global now

    # ensures that 0.5 seconds pass between each prediction in order to reduce errors
    start = datetime.datetime.now().microsecond / 1000
    elapsed = (start - now)

    if guessGesture and abs(elapsed) >= 500:
        t = threading.Thread(target=prediction, args=[frame])
        t.start()
        # prediction(model, frame)
        now = datetime.datetime.now().microsecond / 1000


def prediction(image):
    global pastPrediction, predicting, fullPred, globalModel

    # predictionImg = model.predict(np.array([feature_vector(image)]))
    # accuracy = model.predict_proba(np.array([feature_vector(image)]))

    predictionImg = globalModel.predict(
        np.array([np.hstack([moments(image), haralick(image), color_histogram(image), feature_vector(image)])]))
    accuracy = globalModel.predict_proba(
        np.array([np.hstack([moments(image), haralick(image), color_histogram(image), feature_vector(image)])]))
    # print("prediction", predictionImg, " accuracy", accuracy)
    pred = predictionImg[0]

    if (pred != pastPrediction) and (np.array(pd.DataFrame(accuracy).max(1)) >= 0.98) and (pred != 'other'):

        if pred == 'thumbsUp':
            predicting = True
            print('---------- Start Prediction ----------')
            pastPrediction = pred
            return

        elif pred == 'stop':
            predicting = False
            print('---------- Stop Prediction ----------')
            print(" ".join(fullPred))
            fullPred = []
            pastPrediction = pred
            return

        if predicting:
            print(pred)
            fullPred.append(pred)
            print(np.array(pd.DataFrame(accuracy).max(1)))
            pastPrediction = pred
            return


def train():
    global globalModel
    # initialize the image features intensities matrix and labels list
    imageFeature = []
    # print(imageFeature)
    labels = []
    # print(labels)
    globalModel = 0

    # grab the list of images that we'll be describing
    print("[INFO]  images...")
    imagePaths = list(paths.list_images(TRAIN_PATH))

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our path as the format:
        # /path/to/dataset/{class}_{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split("_")[0]

        # # extract only raw pixel intensity
        # pixels = feature_vector(image)
        #
        # # update the images feature and labels matrices respectively
        # imageFeature.append(pixels)
        # labels.append(label)

        #######################################
        # feature extraction
        f_moments = moments(image)
        f_haralick = haralick(image)
        histogram = color_histogram(image)
        pixels = feature_vector(image)

        # update the images feature and labels matrices respectively
        all_features = np.hstack([f_moments, f_haralick, histogram, pixels])
        imageFeature.append(all_features)
        labels.append(label)
        #######################################

        # show an update every 50 images
        if i > 0 and i % 50 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # show some information on the memory consumed by the raw images matrix
    imageFeature = np.array(imageFeature)
    labels = np.array(labels)

    print("[INFO] Pixels matrix: {:.2f}MB".format(imageFeature.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 90%
    # of the data for training and the remaining 10% for testing
    (trainRawImg, testRawImg, trainRawLabel, testRawLabel) = train_test_split(imageFeature, labels,
                                                                              test_size=TEST_SIZE,
                                                                              random_state=0)

    print("[STATUS] split train and test data...")
    print("Train data  : {}".format(trainRawImg.shape))
    print("Test data   : {}".format(testRawImg.shape))
    print("Train labels: {}".format(trainRawLabel.shape))
    print("Test labels : {}".format(testRawLabel.shape))

    # train and evaluate a k-NN model on the raw pixel intensities
    print("[INFO] Evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=NEIGHBOURS, n_jobs=JOBS)
    model.fit(trainRawImg, trainRawLabel)
    acc = model.score(testRawImg, testRawLabel)
    print("[INFO] Raw pixel accuracy: {:.2f}%".format(acc * 100))
    globalModel = model


def saveRGBImg(image):
    global counter, gestureName, saveImage
    if counter > (NUMOFSAMPLES - 1):
        # Reset parameters
        saveImage = False
        gestureName = ''
        counter = 0
        return

    counter += 1
    img_name = gestureName + "_{}.jpg".format(counter)
    cv2.imwrite(TRAIN_PATH + "/" + gestureName + "/" + img_name, image)
    print("{} written!".format(img_name))
    time.sleep(0.1)


def main():
    global guessGesture, gestureName, saveImage, now
    print('---------------------- Menu ------------------------')
    print(banner)

    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, IMAGE_SIZE)
        cv2.imshow("Original", frame)

        if ret:
            if guessGesture:
                predict(frame)
            if saveImage:
                saveRGBImg(frame)

        # ############# Keyboard inputs ##################
        key = cv2.waitKey(5) & 0xff

        # Use Esc key to close the program
        if key == 27:
            print("Goodbye!")
            break

        #  Use p key to start gesture predictions via KNN model
        elif key == ord('p'):
            print('---------------------- Prediction ------------------------')
            if globalModel == 0:
                print("Train the model first!")
            else:
                guessGesture = not guessGesture

        elif key == ord('t'):
            print('---------------------- Training ------------------------')
            train()

        elif key == ord('h'):
            print('---------------------- Menu ------------------------')
            print(banner)

        elif key == ord('n'):
            gestureName = input("Enter the gesture folder name: ")
            create_folder("./" + TRAIN_PATH + "/" + gestureName)

        elif key == ord('s'):
            print('---------------------- Saving Image ------------------------')
            # saveImage = not saveImage

            if gestureName != '':
                saveImage = True
            else:
                print("Enter gesture folder name first by pressing 'n'")
                saveImage = False

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
