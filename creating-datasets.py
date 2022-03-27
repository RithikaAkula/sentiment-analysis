from importlib.resources import path
import os
import shutil
import random


path_to_dataset="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset/"
SENTIMENTS = ['ANG','DIS','FEA','HAP','NEU','SAD']

filenames=os.listdir(path_to_dataset+'image/')
random.shuffle(filenames)

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1] # 5953
validation_filenames = filenames[split_1:split_2] #744
test_filenames = filenames[split_2:] #745

# for t in test_filenames:
#     original=path_to_dataset+'image/'+t
#     target=path_to_dataset+'split-dataset/test/'+t
#     shutil.copyfile(original, target)

# for t in validation_filenames:
#     original=path_to_dataset+'image/'+t
#     target=path_to_dataset+'split-dataset/validation/'+t
#     shutil.copyfile(original, target)

# for t in train_filenames:
#     original=path_to_dataset+'image/'+t
#     target=path_to_dataset+'split-dataset/train/'+t
#     shutil.copyfile(original, target)


