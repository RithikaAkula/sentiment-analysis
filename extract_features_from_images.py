import os
import numpy as np
import cv2
import pandas as pd


path_to_dataset="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset/"
SENTIMENTS = ['ANG','DIS','FEA','HAP','NEU','SAD']
IMG_SIZE = 277 # as required by AlexNet


def create_dataset(img_folder):
    img_data_array=[]
    class_ids=[]

    counter=0
    print(f"Processing Image {counter+1}\n")

    for file in os.listdir(img_folder):

        filename=file[:-4]
        class_name=filename.split("_")[2]
        class_id=SENTIMENTS.index(class_name)

        image_path = os.path.join(img_folder, file)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 

        img_data_array.append(image)
        class_ids.append(class_id)
        
        counter+=1

    return np.array(img_data_array), np.array(class_ids)



img_folder=path_to_dataset+'image/'
x_images, y_images = create_dataset(img_folder)

print("Processing Done\n")

print("Saving X features file\n")

image_features_x=pd.DataFrame(x_images)
image_features_x.to_csv(path_to_dataset+'image_features_x.csv')

print("Saving Y features file\n")

image_features_y=pd.DataFrame(y_images)
image_features_y.to_csv(path_to_dataset+'image_features_y.csv')

print("END")


