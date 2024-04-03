**Investigating the efficacy of Semi-Supervisedlearning to generate pseudo labels using US VoterFraud data**  

In this project, I classify youtube videos into voter fraud and videos that contain information about the US elections/voterfraud using a novel technique similar to pseudo-labeling. Pseudo labelling uses semi-supervised learning such as k-means clustering. First, on the pre-preprocessed voter fraud dataset, I generated word embeddings using different sentence transformer models. Then, I applied k-means clustering using word-embeddings to generate 200 clusters. To automatically label the clusters, I used pseudo labeling. To test how effectively our clusters were labeled, I trained our data on neural-based state-of-the-art transformer models such as BERT and RoBERTa. The best model (Roberta) achieved up to 83\% Accuracy-Precision Area (AUPRC).

Run in `code` folder:
- Clean dataset: `python video_cleaner.py`
- Sentence embeddings: `python video_vectorizer.py`
- Cluster embeddings: `python video_cluster.py`
- Evaluate embeddings: `python cluster_evaluator.py`.
- Generate Pseudo Labels: `pseduo_labels.py` and `remove_cluster_from_labels.py`
- Generate Graphs: `plot_graphs.py`

