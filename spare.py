import numpy as np
import pandas as pd
path_to_dataset="D:/4-2/MAJOR_PROJECT/sentiment-analysis/"


# LOADING np array file
trainAttr=np.load(path_to_dataset+'audio_np_array.npy')
print(trainAttr.shape)
print(type(trainAttr))

trainY_num = trainAttr[:,[-1]]

trainAttrX = np.delete(trainAttr, -1, axis=1)

print(trainY_num.shape)
print(trainAttrX.shape)
