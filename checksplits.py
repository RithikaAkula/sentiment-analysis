import os

path_to_splits="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset/newsplits/"

trainfiles = os.listdir(path_to_splits+'train/')
print(len(trainfiles)) # 5953

testfiles = os.listdir(path_to_splits+'test/')
print(len(testfiles)) # 745

validationfiles = os.listdir(path_to_splits+'validation/')
print(len(validationfiles)) # 744

# for i in validationfiles: ==> no repeat.
#     if '(1)' in i:
#         print(i)

# for i in validationfiles: ==> no overlap.
#     if i in testfiles:
#         print(i)