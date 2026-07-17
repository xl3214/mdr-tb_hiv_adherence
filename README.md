# mdr-tb_hiv_adherence
This repository contains scripts and de-identified monthly counseling session notes data from the PRAXIS study. 

All analyses were done in Python 3.12.

## Workflow Diagram
```
Raw Counseling Session Notes (not included in this repo)
    ↓
Data Preprocessing (preprocessing.py) — operates on raw notes, not reproducible from shared data
    ↓
BERTopic (bertopic.py)
    ↓
Semantic Similarity Search (semantic_search.py)
    ↓
    └─→ K-Means Clustering of Similarity Score (kmeans_cluster.py)
            ↓
Data Merging + Adherence Intervals
    ↓
Modeling Input Files  ← reproduction starts here with shared data
    ↓
    └─→ Evaluate Models (model_selection.py)
            ↓
Train Linear Mixed-Effects (lme_modeling.py)
            ↓
Assess Effect Modification of Time-Chunk to Cluster (lme_secondary_interaction.py)
            ↓
Interpretation
```

## Data File Directory
1. `query_embeddings.csv`: SBERT embedding file
2. `semantic_search_results.csv`: Semantic similarity search with query text, matched top-5 corpus texts, and similarity score; output of `semantic_search.py` and input of `kmeans_cluster.py`
3. `clustered_queries_by_similarity.csv`: K-means cluster assignment; output of `kmeans_cluster.py`
4. `semantic_similarity_top5_{ART/BDQ}adherence.csv`: Modeling-ready dataset merging top-5 similarity scores, K-means cluster assignment, time-chunk, and adherence outcome, separately for ART and BDQ; input for predictive modeling (`model_selection.py`, `lme_modeling.py`, and `lme_secondary_interaction.py`)

**Note on `Query Text`:** shared text has been preprocessed (lowercased; non-alphabetic characters, stopwords, and short tokens removed; lemmatized), so it reads as a string of content words rather than full sentences. Patient names and gender pronouns were manually removed prior to preprocessing.

## Variable Descriptions

### `semantic_similarity_top5_{ART/BDQ}adherence.csv`

| Variable | Description |
|---|---|
| `PARTICIPANTID` | Participant identifier |
| `VISITCODE` | Study visit code |
| `VISITDATE` | Date of visit |
| `row_id` | Row identifier |
| `Query Text` | Preprocessed counseling note text (see note above) |
| `Cluster` | K-means cluster assignment |
| `top1_Matched Sentence` – `top5_Matched Sentence` | Top 5 corpus sentences ranked by semantic similarity to the query |
| `top1_Similarity Score` – `top5_Similarity Score` | Cosine similarity score for each matched sentence |
| `Chunk` | Time-chunk indicator relative to counseling session |
| `Adherence` | Adherence outcome (ART or BDQ, per filename) |
| `Start_Date` | Start date of the adherence interval |
| `End_Date` | End date of the adherence interval |

### `semantic_search_results.csv`

| Variable | Description |
|---|---|
| `Query Index` | Index of the query note |
| `Query Text` | Preprocessed counseling note text |
| `Rank` | Rank of the matched sentence (1–5) by similarity |
| `Matched Sentence` | Corpus sentence matched to the query |
| `Similarity Score` | Cosine similarity score between query and matched sentence |

### `query_embeddings.csv`

| Variable | Description |
|---|---|
| `dim_0` – `dim_383` | SBERT embedding dimensions for each query |

### `clustered_queries_by_similarity.csv`

| Variable | Description |
|---|---|
| `Query Text` | Preprocessed counseling note text |
| `Cluster` | K-means cluster assignment |
