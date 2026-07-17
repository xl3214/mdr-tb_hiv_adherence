# mdr-tb_hiv_adherence
This repository contains scripts and de-identified monthly counseling session notes data from the PRAXIS study. 

All analyses were done in Python 3.12.

## Data File Directory
1. `query_embeddings.csv`: SBERT embedding file
2. `semantic_search_results.csv`: Semantic similarity search with query text, matched top-5 corpus texts, and similarity score; output of `semantic_search.py` and input of `kmeans_cluster.py`
3. `clustered_queries_by_similarity.csv`: K-means cluster assignment; output of `kmeans_cluster.py`
4. `semantic_similarity_top5_{ART/BDQ}adherence.csv`: Modeling-ready dataset merging top-5 similarity scores, K-means cluster assignment, time-chunk, and adherence outcome, separately for ART and BDQ; input for predictive modeling (`model_selection.py`, `lme_modeling.py`, and `lme_secondary_interaction.py`)

**Note on `Query Text`:** shared text has been preprocessed (lowercased; non-alphabetic characters, stopwords, and short tokens removed; lemmatized), so it reads as a string of content words rather than full sentences. Patient names and gender pronouns were manually removed prior to preprocessing.

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
