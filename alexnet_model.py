import keras
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import cv2
from numpy import *
from keras.utils import np_utils


'''
The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):

Anger (ANG)
Disgust (DIS)
Fear (FEA)
Happy/Joy (HAP)
Neutral (NEU)
Sad (SAD)
and emotion level (in parentheses is the two letter code used in the fourth part of the filename):

Low (LO)
Medium (MD)
High (HI)
Unspecified (XX)
The suffix of the filename is based on the type of file, flv for flash video used for presentation of both the video only, and the audio-visual clips. mp3 is used for the audio files used for the audio-only presentation of the clips. wav is used for files used for computational audio processing.

'''

path_to_splits="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset/split-dataset/"
SENTIMENTS = ['ANG','DIS','FEA','HAP','NEU','SAD']
IMG_SIZE=277 # as required by AlexNet

''' CREATING TensorFlow Dataset representations '''
def create_dataset(img_folder):
    img_data_array=[]
    class_ids=[]
    for file in os.listdir(img_folder):

        filename=file[:-4]
        class_name=filename.split("_")[2]
        class_id=SENTIMENTS.index(class_name)

        image_path= os.path.join(img_folder, file)
        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (IMG_SIZE, IMG_SIZE),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 

        img_data_array.append(image)
        class_ids.append(class_id)

    return img_data_array, class_ids


# extract the image array and class name
train_images, train_labels = create_dataset(path_to_splits+'train/')
test_images, test_labels = create_dataset(path_to_splits+'test/')
validation_images, validation_labels = create_dataset(path_to_splits+'validation/')

''' converting labels to categorical data '''
train_labels = np_utils.to_categorical(train_labels, 6)
test_labels = np_utils.to_categorical(test_labels, 6)
validation_labels = np_utils.to_categorical(validation_labels, 6)

# print(Y[100]) # [0. 0. 1. 0. 0. 0.]
# print(shape(Y)) # (745, 6)


''' Converting to TensorFlow Dataset representation '''
def convert_to_tfds(images_array, labels):
    return tf.data.Dataset.from_tensor_slices((images_array, labels))


train_ds = convert_to_tfds(train_images, train_labels)
test_ds = convert_to_tfds(test_images, test_labels)
validation_ds = convert_to_tfds(validation_images, validation_labels)


''' Get the size of dataset partitions '''
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)


''' PIPELINE to shuffle and batch the datasets '''
train_ds = (train_ds
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

''' ALEX-NET MODEL IMPLEMENTATION '''
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


''' TENSORBOARD '''
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

''' COMPILE THE MODEL '''
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()


'''' TRAIN THE NETWORK '''
def train_model(model):
    model.fit(train_ds,
            epochs=50,
            validation_data=validation_ds,
            validation_freq=1,
            callbacks=[tensorboard_cb])

train_model(model)


''' EVALUATE THE MODEL '''
model.evaluate(test_ds)