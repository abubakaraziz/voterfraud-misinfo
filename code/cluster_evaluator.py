import os, json
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score,rand_score,v_measure_score
from experiment_settings import event, model_names


# Evaluation metrics.
metric_names = ["Mutual info", "Homogeneity", "Completeness","Randscore","Vscore"]
metrics = [adjusted_mutual_info_score, homogeneity_score, completeness_score,rand_score,v_measure_score]


# Read data.
clusters_folder = os.path.join("..", "data", event, "clusters")
for model_name in model_names:
    print("\n", model_name)
    clusters_path = os.path.join(clusters_folder, model_name + ".csv")
    data = pd.read_csv(clusters_path,nrows=50)
    
    # Evaluate clusters tweet-based pseudo-labels.
    labels_true = data["community_label"]
    labels_pred = data["cluster"]
    for metric_name, metric in zip(metric_names, metrics):
        metric_eval = metric(labels_true, labels_pred)
        print("\t", metric_name, "\t", metric_eval)

        # Print some some samples.
    #for cluster, g in data.groupby("cluster"):
       # print(cluster)
        #for text in g["text"][:4]:
        #    print(text)



