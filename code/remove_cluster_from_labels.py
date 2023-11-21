import os, json
import pandas as pd
from experiment_settings import event
import numpy as np

 
distance = "euclidean_distance"
file_name="data"
train = "train"
test = "test"

#Read distance data

distance_folder = os.path.join("..","data",event,"distance")
distance_path = os.path.join(distance_folder,distance+".csv")
all_pseudo_labels = pd.read_csv(distance_path)

# Read Labelled Data 
distance_folder=os.path.join("..","data",event,"label")
distance_path=os.path.join(distance_folder,file_name+".csv")
labelled_data=pd.read_csv(distance_path)

cond = labelled_data['text'].isin(all_pseudo_labels['text'])


train_data = labelled_data.drop(labelled_data[cond].index)
test_data = pd.merge(all_pseudo_labels, labelled_data,on=["text"], how="inner")
#delete psuedo-label columns from test set which came from euclidean_distance
del test_data['label']
del test_data['cluster']
print(train_data)
print(test_data)

#Saving Test and Train Data 
label_folder = os.path.join("..","data",event,"label")
train_path = os.path.join(label_folder,train+".csv")

test_path = os.path.join(label_folder, test+".csv")

train_data.to_csv(train_path)
test_data.to_csv(test_path)
