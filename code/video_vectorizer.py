import os, json
import pandas as pd
from sentence_transformers import SentenceTransformer
from experiment_settings import event, model_names


# Read data.
data_path = os.path.join("..", "data", event, "clean.csv")
data = pd.read_csv(data_path)


# Experiment for each model.
embedding_folder = os.path.join("..", "data", event, "embeddings")
os.makedirs(embedding_folder, exist_ok=True)
for model_name in model_names:
    embeddings_path = os.path.join(embedding_folder, model_name + ".csv")
    if (not os.path.isfile(embeddings_path)):
        # Load the pre-trained model.
        print("Loading : ",model_name)
        model = SentenceTransformer(model_name)
        print("Model loaded:", model_name)


        # Vectorize corpus.
        corpus = data["text"].tolist()
        embeddings = model.encode(corpus)


        # Save embeddings.
        embeddings = pd.DataFrame(embeddings)
        embeddings.to_csv(embeddings_path, index=False)
        print("Embeddings exported:", model_name)
