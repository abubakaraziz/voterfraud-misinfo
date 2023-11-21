import os, json
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from experiment_settings import event, model_names, cluster_num


# Read data.
data_path = os.path.join("..", "data", event, "clean.csv")
data = pd.read_csv(data_path)
corpus = data["text"].tolist()


# Read embeddings.
embedding_folder = os.path.join("..", "data", event, "embeddings")
clusters_folder = os.path.join("..", "data", event, "clusters")
os.makedirs(clusters_folder, exist_ok=True)
for model_name in model_names:
    clusters_path = os.path.join(clusters_folder, model_name + ".csv")
    if (not os.path.isfile(clusters_path)):
        embeddings_path = os.path.join(embedding_folder, model_name + ".csv")
        embeddings = pd.read_csv(embeddings_path).to_numpy()


        # Cluster embeddings.
        clustering_model = KMeans(n_clusters=cluster_num)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        
        
        # Save clusters.
        data["cluster"] = cluster_assignment
        clusters_path = os.path.join(clusters_folder, model_name + ".csv")
        data.to_csv(clusters_path, index=False)
        print("Clusters exported:", model_name)
