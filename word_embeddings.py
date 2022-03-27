'''

TEXT IN DATASET:
It's eleven o'clock (IEO).
That is exactly what happened (TIE).
I'm on my way to the meeting (IOM).
I wonder what this is about (IWW).
The airplane is almost full (TAI).
Maybe tomorrow it will be cold (MTI).
I would like a new alarm clock (IWL).
I think I have a doctor's appointment (ITH).
Don't forget a jacket (DFA).
I think I've seen this before (ITS).
The surface is slick (TSI).
We'll stop in a couple of minutes (WSI).


'''

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot


# access file contaning sentences
sentences_file=open("D:/4-2/MAJOR_PROJECT/sentiment-analysis/sentences.txt").read()
lines=sentences_file.split("\n")
labels = np.array([0,0,1,1,0,0,1,1,0,1,0,0])

# integer encode the sentences
vocab_size = 50
# Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding.
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/one_hot
encoded_docs = [one_hot(line, vocab_size) for line in lines]
print(encoded_docs)

# pad sequences
# pad documents to a max length of 6 words
max_length = 6
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
model = Sequential()
'''
The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:
It must specify 3 arguments:
input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.
'''
model.add(Embedding(vocab_size, 12, input_length=max_length))
# the output from the Embedding layer will be 6 vectors of 12 dimensions each, one for each word. We flatten this to a one 72-element vector to pass on to the Dense output layer.
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))