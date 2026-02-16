# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "bertopic>=0.17.4",
#     "natsort>=8.4.0",
#     "pandas>=3.0.0",
# ]
# ///

import os
import sys

# loky/numba parallelism causes resource tracker errors on macOS
if sys.platform == "darwin":
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

import re
import glob
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from natsort import natsorted

# to clearly differentiate between the script logs and the BERT logs
def log(message):
    print(f">>> {message}")

CACHE_DIR = Path(".cache")
DOCS_CACHE = CACHE_DIR / "docs.joblib"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.joblib"

parser = argparse.ArgumentParser(description="Run BERTopic on far-right Telegram data.")
parser.add_argument(
    "--no-cache",
    action="store_true",
    help="Ignore cached documents and embeddings, recompute everything from scratch.",
)
args = parser.parse_args()
use_cache = not args.no_cache

# check for cached data
cached_data = None
if not use_cache:
    log("Cache disabled via --no-cache flag.")
elif not DOCS_CACHE.exists() or not EMBEDDINGS_CACHE.exists():
    log(f"No cache found in {CACHE_DIR}/")
else:
    log(f"Loading cached documents from {DOCS_CACHE}...")
    docs = joblib.load(DOCS_CACHE)
    log(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
    embeddings = joblib.load(EMBEDDINGS_CACHE)
    log(f"Loaded {len(docs)} documents and embeddings from cache.")
    cached_data = (docs, embeddings)

if cached_data is None:
    # collect and process documents
    clubnames = r"./docs/network-channel-names.csv"
    log(f"Reading clubnames from {clubnames}...")
    clubnames_df = pd.read_csv(clubnames)
    clubnames_list = clubnames_df["clubname"].dropna().tolist()
    log(f"Loaded {len(clubnames_list)} club names.")

    datasets_chunked = natsorted(glob.glob(r"./data/processed_part_*.csv"))
    log(f"Found {len(datasets_chunked)} data files (sorted numerically).")

    channelnames = []

    log("Collecting channel names from data files...")
    for file in datasets_chunked:
        df = pd.read_csv(file, usecols=["channel_name"])
        channelnames.extend(df["channel_name"].dropna().unique())

    channelname_df = pd.DataFrame({"channel_name": list(set(channelnames))})
    log(f"Found {len(channelname_df)} unique channel names.")
    save_path = "./output/allchannelnames.csv"
    channelname_df.to_csv(save_path, index=False, encoding="utf-8")
    log(f"Saved channel names to {save_path}")

    channelnames_list = channelname_df["channel_name"].tolist()
    channelnames_pattern = r'\b(?:' + '|'.join(map(re.escape, channelnames_list)) + r')\b'

    docs = []
    unique_docs = set()

    log("Filtering and deduplicating documents...")
    total = len(datasets_chunked)
    for i, dataset in enumerate(datasets_chunked, 1):
        log(f"  [{i}/{total}] {os.path.basename(dataset)}")
        try:
            df_iter = pd.read_csv(dataset, usecols=["channel_name", "cleaned_message"],
                                  sep=None, engine="python", encoding="utf-8", chunksize=10000)

            for chunk in df_iter:
                filtered_chunk = chunk[chunk["channel_name"].isin(clubnames_df["clubname"])].copy()
                filtered_chunk = filtered_chunk.drop_duplicates(subset=["cleaned_message"])
                filtered_chunk.loc[:, "cleaned_message"] = filtered_chunk["cleaned_message"].str.replace(channelnames_pattern, '', regex=True)

                for msg in filtered_chunk["cleaned_message"].dropna():
                    if msg not in unique_docs:
                        unique_docs.add(msg)
                        docs.append(msg)

        except Exception as e:
            log(f"  Error processing {dataset}: {e}")

    log(f"Total unique documents collected: {len(docs)}")

    # compute embeddings
    from sentence_transformers import SentenceTransformer

    log("Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    log(f"Encoding {len(docs)} documents (this may take a while)...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    log("Encoding complete.")

    # save cache
    CACHE_DIR.mkdir(exist_ok=True)
    log(f"Saving documents to {DOCS_CACHE}...")
    joblib.dump(docs, DOCS_CACHE)
    log(f"Saving embeddings to {EMBEDDINGS_CACHE}...")
    joblib.dump(embeddings, EMBEDDINGS_CACHE)
    log("Cache saved.")

else:
    docs, embeddings = cached_data

# run topic model
log("Loading dependencies...")
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
log("Dependencies loaded.")

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

log("Initializing UMAP, and CountVectorizer...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

log("Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    # we use the default HDBSCAN model from BERTopic
    vectorizer_model=vectorizer_model,
    top_n_words=5,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings)
n_topics = len(set(topics)) - (1 if -1 in topics else 0)
log(f"BERTopic fitting complete. Found {n_topics} topics.")

# reduced embeddings for visualization, n_jobs=1 to avoid libomp segfault on macOS ARM64
log("Computing reduced embeddings for visualization...")
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', n_jobs=1).fit_transform(embeddings)
log("Reduced embeddings complete.")

log(f"Docs: {len(docs)} | Topics: {len(topics)} | Model topics: {len(topic_model.topics_)}")

# save per-document results
log("Saving results...")
df = pd.DataFrame({"topic": topics, "document": docs})
save_path_df = "./output/topic-model-results.csv"
df.to_csv(save_path_df, index=False, encoding="utf-8")
log(f"Results saved to {save_path_df}")

# save topic overview (id, count, name, top words)
topic_info = topic_model.get_topic_info()
save_path_topics = "./output/topic-info.csv"
topic_info.to_csv(save_path_topics, index=False, encoding="utf-8")
log(f"Topic info saved to {save_path_topics}")

log("Start visualizing top 100 topics")
# Visualize topics
fig = topic_model.visualize_topics(topics = list(range(0, 100)))
fig.write_html("./output/visualized_topics.html")
fig.write_image("./output/visualized_topics.png")
log("Finished visualizing top 100 topics")

log("Generating barchart of top 100 topics")
# Generate barchart of top keywords
figbar = topic_model.visualize_barchart(list(range(0, 100)))
figbar.write_html("./output/barchart.html")
figbar.write_image("./output/barchart.png")
log("Finished generating barchart of top 100 topics")

log("Generating hierarchical clustering map of top 100 topics")
# Generate hierarchical clustering map
hierarchy = topic_model.visualize_hierarchy(orientation="left", top_n_topics=100)
hierarchy.write_html("./output/visualized_hierarchy.html")
hierarchy.write_image("./output/visualized_hierarchy.png")
log("Finished generating hierarchical clustering map of top 100 topics")

log("Generating document map of top 100 topics")
# Generate document map visualization
fig = topic_model.visualize_documents(
    docs=docs,
    reduced_embeddings=reduced_embeddings,
    topics=list(range(0, 100))
)
fig.write_image("./output/visualization_map.png")
fig.write_html("./output/visualization_map.html")
log("Finished generating document map of top 100 topics")

log("Done.")