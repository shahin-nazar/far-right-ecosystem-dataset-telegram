# Python 3.x

import pandas as pd
import csv
import glob
import collections
import re

from bertopic import BERTopic

clubnames = r"./docs/network-channel-names.csv"
print(r"Reading clubnames...")
clubnames_df = pd.read_csv(clubnames)
clubnames_list = clubnames_df["clubname"].dropna().tolist()

docs = []

for dataset, idx in x in enumerate(datasets_chunked):
    try:
        print(r"Processing dataset {idx}")
        # read file in chunks to avoid memory issues
        df_iter = pd.read_csv(dataset, usecols=["channel_name", "cleaned_message"], 
                              sep=None, engine="python", encoding="utf-8", chunksize=10000)

        for chunk in df_iter:
            # Filter rows where channel_name is in clubnames
            filtered_chunk = chunk[chunk["channel_name"].isin(clubnames_df["clubname"])]

            # Remove words in cleaned_message that match any entry in channelname_df
            filtered_chunk.loc[:,"cleaned_message"] = filtered_chunk["cleaned_message"].str.replace(channelnames_pattern, '', regex=True)

            # Add filtered cleaned_message entries to docs
            docs.extend(filtered_chunk["cleaned_message"].dropna().tolist())

    except Exception as e:
        print(f"Error processing {dataset}:{e}")

print(f"Total documents collected: {len(docs)}") # check total messages collected


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = embedding_model.encode(docs, show_progress_bar=True)

from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    top_n_words=20,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)

reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

print("Docs length:", len(docs))
print("Topics length:", len(topics))

print("Model topics_ length:", len(topic_model.topics_))

df = pd.DataFrame({"topic": topics, "document": docs})


save_path_df = "./output"
df.to_csv(save_path_df, index=False, encoding="utf-8")