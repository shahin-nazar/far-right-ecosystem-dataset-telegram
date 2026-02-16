[![DOI](https://zenodo.org/badge/1157131514.svg)](https://doi.org/10.5281/zenodo.18633027)

# The Far-Right Telegram Ecosystem Dataset (1025 Channels, 5.7M+ Posts)

This repository contains public Telegram data of 1025 far-right groups used to apply BERT-based topic modeling. The dataset contains over 5.7 million public posts posted between 2019-2024.

The repo includes processed data, BERT model example outputs, and other materials used in the corresponding paper (to be published).

The repository is organized into several directories containing various components of the project. Below is a detailed description of the repository structure, usage, and how to cite this work.

## Repository Structure

- `./`: For easy exploration of the dataset's themes or reproducability of the associated paper: run-model.py runs BERTopic on a specific subset of channels in the data in `/data` and stores the output in `/output`.
- `data/`: Contains the processed data, which includes chunked-up CSV files ready for analysis.
- `docs/`: Contains: a CSV file with all channel names of the dataset, a CSV file with Active Club channel names used to filter dataset and process with BERTopic.
- `model-output/`: Contains the output of the topic model in CSV format, including a few example visualizations of the output.

## Running the Model

`run-model.py` (and the equivalent `run-model.ipynb` notebook) runs BERTopic on a filtered subset of channels. It reads the data from `data/`, filters by channel names in `docs/network-channel-names.csv`, fits a topic model, and writes the results to `output/`.

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. Install it by following the instructions at https://docs.astral.sh/uv/getting-started/installation/. You do not need to install Python or any packages yourself. `uv` handles all of that automatically based on the `pyproject.toml` in this repository.

### Running the script

From the root of this repository, run:

```bash
uv run run-model.py
```

On the first run, `uv` will download the correct Python version and install all required packages. Subsequent runs reuse the cached environment. Output files are written to `output/`.

The script caches processed documents and embeddings in `.cache/` to avoid recomputing them on subsequent runs. To force a fresh run (e.g., if you changed the input data), use:

```bash
uv run run-model.py --no-cache
```

## Dataset

The dataset used in this project is obtained from public Telegram channels and groups and consists of messages and interactions, which were processed into CSV format.

This data was collected using [telegram-tracker](https://github.com/estebanpdl/telegram-tracker).

Includes a broad array of far-right movements and groups such as: 
- White supremacists, ultranationalists, identitarians, neo-Nazis, esoteric Nazism, Christian Nationalists, accelerationists of different colors, great replacement thinking and other conspiracy theories, militias, ecofascists, tradwives, and many more. 
- Also includes contemporary movements such as the Active Clubs, Atomwaffen related groups, Junge Nationalisaten, Patriot Front, Oath Keepers, Nordic Resistance Movement, Patriotic Alternative, Patriot Movement, and, indeed, many more.
- Thinktanks such as the White Papers Institute.
- Right-wing influencers and platforms such as Red Ice TV, Joel Davis, MEDIA2RISE, etc.

Post-level data and various fields including: channel ID's, channel names, message ID's, message texts, engagement metrics (views, number of replies, etc.), forwards, message links, media type contained, domain, url, and about a dozen more fields.

The data was obtained via the Telegram API using iterative snowball sampling forwarded accounts of a seed group, thus capturing the relational and networked dimensions of the online far-right ecosystem.

### Data Description:

- The processed data is stored in the `data/` directory. It includes 571 CSV files each with 10,000 rows that have been chunked to fit into BERT-based topic models. The data is preprocessed and thus easily parseable with Pandas.
- Each row is a post and contains the following fields/colums:
  - `signature`
  - `channel_id`
  - `channel_name`
  - `msg_id`
  - `message`
  - `cleaned_message`
  - `date`
  - `msg_link`
  - `msg_from_peer`
  - `msg_from_id`
  - `views`
  - `number_replies`
  - `number_forwards`
  - `is_forward`
  - `forward_msg_from_peer_type`
  - `forward_msg_from_peer_id`
  - `forward_msg_from_peer_name`
  - `forward_msg_date`
  - `forward_msg_date_string`
  - `forward_msg_link`
  - `is_reply`
  - `reply_to_msg_id`
  - `reply_msg_link`
  - `contains_media`
  - `media_type`
  - `has_url`
  - `url`
  - `domain`
  - `url_title`
  - `url_description`
  - `document_type`
  - `document_id`
  - `document_video_duration`
  - `document_filename`
  - `poll_id`
  - `poll_question`
  - `poll_total_voters`
  - `poll_results`
  - `contact_phone_number`
  - `contact_name`
  - `contact_userid`
  - `geo_type`
  - `lat`
  - `lng`
  - `venue_id`
  - `venue_type`
  - `venue_title`
  - `venue_address`
  - `venue_provider`

We ensured respect for Telegram's [Terms of Service](https://telegram.org/tos) and data privacy laws (e.g., GDPR, CCPA) when working with this dataset.

### Model Settings

The BERT-based topic modeling pipeline includes the following settings:

1. **CountVectorizer**:

   - **stop_words**: 'english' (removes common English stopwords)
   - **min_df**: 2 (ignores terms that appear in fewer than 2 documents)
   - **ngram_range**: (1, 2) (uses unigrams and bigrams)

2. **UMAP** (for dimensionality reduction):

   - **n_neighbors**: 15 (number of neighboring points used for manifold approximation)
   - **n_components**: 5 (reduces the data to 5 components for the clustering)
   - **min_dist**: 0.0 (controls how tightly UMAP packs points together)
   - **metric**: 'cosine' (distance metric for UMAP)

3. **BERTopic** (the main topic model):

   - **embedding_model**: Your BERT embedding model (e.g., `bert-base-uncased`)
   - **umap_model**: UMAP model used for dimensionality reduction
   - **vectorizer_model**: CountVectorizer model used for tokenization
   - **top_n_words**: 20 (number of top words to display for each topic)
   - **verbose**: True (provides detailed output during the topic modeling process)

4. **UMAP for Reduced Embeddings**:
   - **n_neighbors**: 10 (number of neighboring points for the reduced space)
   - **n_components**: 2 (reduces the embeddings to 2 components for visualization)
   - **min_dist**: 0.0 (controls how tightly UMAP packs points together)
   - **metric**: 'cosine' (distance metric for UMAP)

# Example of reduced embeddings visualization

```python
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
```

## Results

The topic modeling results are stored in the `model-output/` folder in CSV format.

## Figures

The visualizations of the BERT model's output, including topic distributions and other figures used in the associated paper, are stored in the `figures/` folder. Images from Telegram are in JPG format and can be directly used in publications or presentations.

# Data Privacy and Anonymization

This dataset contains public data scraped from public Telegram channels and groups. While all content in the dataset is publicly available, it may contain personal information, including email addresses or contact information that users have voluntarily shared in their public posts.

In accordance with privacy best practices, we have done our best to anonymize the dataset by removing or replacing sensitive information (e.g., email addresses). However, please note that users' publicly shared content is still present and that we may have missed something.

Disclaimer: No private conversations or data from private groups have been included in this dataset. The data is sourced only from publicly accessible Telegram groups/channels.

By using this dataset, you agree to respect the privacy of individuals and adhere to the relevant privacy laws and regulations.

## License

This repository is licensed under the Creative Commons Attribution 4.0 International Share-Alike (CC BY-SA 4.0).

The dataset is obtained from public Telegram channels and groups. While we provide the data in this repository, please ensure you follow Telegram's Terms of Service and data privacy laws when using this dataset.

## Citation

If you use this dataset or model in your research, please cite the following:

```bibtex
@misc{nazar2026,
  author = {Shahin Nazar, Thomas F. K. Jorna, Abigail Nieves Delgado, Toine Pieters},
  title = {"The Far-Right Telegram Ecosystem Dataset (1025 Channels, 5.7M+ Posts)"},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18633027},
  url = {https://zenodo.org/records/18633027}
}
```
