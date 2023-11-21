import os, json
import pandas as pd
from experiment_settings import event
import numpy as np



def same_label_clusters(value):
    if value[0]==value[1]:
        return 1
    else:
        return 0

def check_same_labels(group, label):
    same_label = 1 if group['label'][0] == group['label'][1] == label else 0
    return same_label
def check_different_clusters(group): 
    different_label = 1 if group['label'][0] != group['label'][1] else 0
    return different_label


#Read cluster data
clusters_folder=os.path.join("..","data",event,"clusters")
model_name="paraphrase-distilroberta-base-v1"
cluster_path=os.path.join(clusters_folder,model_name+".csv")
all_data=pd.read_csv(cluster_path)
#print(all_data)
# Read cluster_labels_data from distance metric
distance_metric="euclidean_distance_complete"
distance_folder=os.path.join("..","data",event,"distance")
distance_path=os.path.join(distance_folder,distance_metric+".csv")
labelled_data=pd.read_csv(distance_path)
clusters=labelled_data.groupby('cluster')

same_clusters=(clusters.label.agg(same_label_clusters).to_frame())
print("Clusters with Same Labels are", same_clusters[same_clusters["label"] == 1.0].sum())

#Get different cluster statistics

print("Clusters with different Labels are", len(same_clusters[same_clusters["label"] == 0.0]))


promote_voterfraud_cluster=[] 
rebut_voterfraud_cluster=[]
information_voter_cluster=[]
other_cluster=[]
different_label_cluster=[]
for name, group in clusters:
    #counting total number of labels claiming voterfraud
    label = 1
    same_label = check_same_labels(group, label)
    promote_voterfraud_cluster.append(name) if same_label == 1 else 0
    
    #counting total number of labels rebutting voterfraud
    
    label = 2
    same_label = check_same_labels(group, label)
    rebut_voterfraud_cluster.append(name) if same_label == 1 else 0
    

    #counting total number of labels containing information about US elections 
    
    label =  3
    same_label = check_same_labels(group, label)
    information_voter_cluster.append(name) if same_label == 1 else 0


    #counting total number of labels containing information about US elections 
    
    label = 4
    same_label = check_same_labels(group, label)
    other_cluster.append(name) if same_label == 1 else 0

    #counting clusters with different labels
    different_label = check_different_clusters(group)
    different_label_cluster.append(name) if different_label == 1 else 0 


print("Total Clusters Promoting VoterFraud {}".format(len(promote_voterfraud_cluster)))
print("Total Clusters Rebut VoterFraud {}".format(len(rebut_voterfraud_cluster)))
print("Total Clusters Containing information about Elections {}".format(len(information_voter_cluster)))
print("Total Clusters Containing Other Information {}".format(len(other_cluster)))

election_and_other_cluster = information_voter_cluster + other_cluster + rebut_voterfraud_cluster

#Adding Label Symbol To pandas DataFrame
all_data['labels']=-2

all_data_group_cluster=all_data.groupby('cluster')

#Clusters containg videos promoting voterfraud information
voterfraud_promote_titles=0
for voterfraud_cluster in (promote_voterfraud_cluster):
    for cluster, group in all_data_group_cluster:
        if cluster == voterfraud_cluster:
            voterfraud_promote_titles=voterfraud_promote_titles+len(group)
            all_data.loc[all_data.cluster == voterfraud_cluster, "labels"] = 0

#Clusters containg other election information
election_titles=0
for election_cluster in (election_and_other_cluster):
    for cluster, group in all_data_group_cluster:
        if cluster == election_cluster:
            election_titles = election_titles+len(group)
            all_data.loc[all_data.cluster == election_cluster, "labels"] = 1 

#Cluster that contains cluster videos different videos

different_titles=0
for election_cluster in (different_label_cluster):
    for cluster, group in all_data_group_cluster:
        if cluster == election_cluster:
            different_titles = different_titles + len(group)


print("Total Election Titles Videos {}".format(election_titles))
print("Total videos promoting Voter Fraud {}".format(voterfraud_promote_titles))
print("Total Videos that didn't agree on any labels {}".format(different_titles))
#print("Total Clusters Promoting VoterFraud {}".format((promote_voterfraud_cluster)))
print(different_label_cluster)

#get only those labels
only_labelled_data=all_data.loc[( all_data['labels'] == 0) | (all_data['labels'] == 1 )]
model_name="data_complete"
labelled_path=os.path.join("..","data",event,"label")
only_labelled_file=os.path.join(labelled_path,model_name+".csv")
only_labelled_data=only_labelled_data[["labels", "text"]]
only_labelled_data.to_csv(only_labelled_file, index=False)
