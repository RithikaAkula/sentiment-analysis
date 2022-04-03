import os
import shutil
import random


path_to_dataset="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset/"

filenames=os.listdir(path_to_dataset+'image/')
random.shuffle(filenames)


split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
# print(len(filenames))
# print(split_1, split_2) # 5953 6697
train_filenames = filenames[:split_1] # 5953
validation_filenames = filenames[split_1:split_2] #744
test_filenames = filenames[split_2:] #745

        
for x in test_filenames:
    original=path_to_dataset+'image/'+x
    target=path_to_dataset+'newsplits/test/'+x
    shutil.copyfile(original, target)

for y in validation_filenames:
    original=path_to_dataset+'image/'+y
    target=path_to_dataset+'newsplits/validation/'+y
    shutil.copyfile(original, target)

for z in train_filenames:
    original=path_to_dataset+'image/'+z
    target=path_to_dataset+'newsplits/train/'+z
    shutil.copyfile(original, target)


