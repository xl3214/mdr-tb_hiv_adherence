# mdr-tb_hiv_adherence
This repository contains scripts and de-identified monthly counseling session notes data from the PRAXIS study. 

All analyses were done in Python 3.12.

# Data File Directory
1. SBERT embedding file - query_embeddings.csv
2. Semantic similarity search with query text, matched top-5 corpus texts, and similarity score - output of semantic_search.py and input of kmeans_cluster.py - semantic_search_results.csv
3. K-means cluster assignment - output of k_means.py - clustered_queries_by_similarity.csv
4. meta-data of top-5 similarity score, k-means cluster assignment, mapped to time-chunk and adherence, separately for BDQ and ART - input for predictive modeling - semantic_similarity_top5_{BDQ/ART}adherence.csv

## Workflow Diagram
```
Data Processing (preprocessing.py)
    ↓  
    └─→ BERTopic (bertopic.py)
            ↓
Semantic Similarity Search (semantic_search.py)
            ↓
            └─→ K Means Clustering of Similarity Score (kmeans_cluster.py)
                    ↓
Data Merging + Adherence Intervals
    ↓
Modeling Input Files
    ↓
    └─→ Evaluate Models (model_selection.py)
            ↓
Train Linear Mixed-Effects (lme_modeling.py)
            ↓
Assess Effect Modification of Time-Chunk to Cluster (lme_secondary_interaction.py)
            ↓
      Interpretation
```
