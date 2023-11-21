import os, json
import pandas as pd
from experiment_settings import event
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



def calculate_euclidean_distance(sentence1,sentence2):
    return euclidean_distances(sentence1,sentence2) 

#Read Data
clusters_folder=os.path.join("..","data",event,"labelled")
model_name="paraphrase-distilroberta-base-v1"
cluster_path=os.path.join(clusters_folder,model_name+".csv")
all_data=pd.read_csv(cluster_path)
print(all_data)
cluster_groups=all_data.groupby('cluster')['text']

key_list_from_cluster_group=[i for i in range(0,200)]

for cluster_key,item in cluster_groups:

    data=None
    if cluster_key in key_list_from_cluster_group:
        data=cluster_groups.get_group(cluster_key)
        data=data.reset_index(drop=True)
               
        vectorizer=CountVectorizer()
        features=vectorizer.fit_transform(data).todense()
        
        #calculating distance between two pair of matrices
        distance_pair_matrix=euclidean_distances(features,features)
        maximum_value=distance_pair_matrix.max()

        max_x=0
        max_y=0
        for x in range(0,len(distance_pair_matrix)):
            for y in range(0,len(distance_pair_matrix)):
                if distance_pair_matrix[x][y]==maximum_value:
                    max_x,max_y=x,y 
        sentence1=data[max_x]
        sentence2=data[max_y]
        #write data
        distance_folder=os.path.join("..","data",event,"distance")
        metric_name="euclidean_distance"
        metric_file_path=os.path.join(distance_folder,metric_name+".csv")
        d={'text':[sentence1,sentence2],'cluster':[cluster_key,cluster_key]}
        df=pd.DataFrame(data=d)
        df.to_csv(metric_file_path,mode='a',header=False)


'''
data=data.to_numpy()


distances=np.zeros((len(data),len(data)))
for x in range(0,len(data)):
    sentence1=data[x]
    print(sentence1)
    for y in range(0,len(data)):
        sentence2=data[y]
        distances[x][y]=calculate_euclidean_distance(sentence1,sentence2)

print(distances)

for f in features:
    print(euclidean_distances(features[0],f))
'''

#print(data.to_frame())
#print("Sentence 1:",sentence1,"Sentence 2:", sentence2)



































