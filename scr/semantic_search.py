'''
Semantic Similarity Search Script
1. Calculate cosine similarity score between session notes and pseudo-semantic corpus
2. Selecting the top-5 similarity scores

Input: Raw texts or preprocessed texts of counseling session notes
Output: semantic_search_results.csv
'''

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocessing2 import load_and_preprocess

# SBERT Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Loading Counseling Session Notes with Preprocessing Function (from "preprocessing.py")
docs = load_and_preprocess("/Users/kristal99/Desktop/DA/PRAXIS/NLP/Rproj/data/mci1_preprocess.xlsx")
# Due to participant identification purposes, the raw texts will not be made available.
# For replication purposes, preprocessed texts are made available ("preprocessed_query_texts.csv")
# The analysis results can be replicated without going through the `load_and_preprocess` function. 

# Pseudo Semantic Corpus, created based on the topics emerged from BERTopic results. 
corpus = [
    "Patient experienced side effects like nausea, fatigue, painful joints and feet, poor vision, and blindness.",
    "Fear of community stigma discouraged patient from attending clinic appointments.",
    "Patient expressed determination to complete TB treatment and stay healthy.",
    "Challenges in remembering doses led to missed medication days this week.",
    "Patient described food insecurity as a barrier to taking medication regularly.",
    "Living in rural area and poor internet connection caused Wisepill device to not record intakes.",
    "Patient disclosed alcohol use interfering with evening medication intake.",
    "Counselor emphasized setting reminders and establishing a routine for medication.",
    "Patient received positive reinforcement for improved adherence this month.",
    "Missed appointments were due to transport issues and clinic wait times.",
    "Patient shared feelings of isolation and lack of emotional support at home.",
    "Mental health concerns, including depression and anxiety, were discussed.",
    "Counselor helped identify treatment goals and steps toward long-term adherence.",
    "Support group sessions were discussed as a source of encouragement.",
    "Referral was made to social worker to address financial and housing needs.",
    "Patient explained confusion about pill timing and dosage instructions.",
    "Motivational interviewing was used to build confidence in treatment plan.",
    "Side effects of ART and TB drugs were clarified and normalized.",
    "Substance use and traditional medicine were discussed.",
    "The session ended with affirming patient’s efforts and planning next steps."
]

# Embedding the Corpus
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
results = []
top_k = min(5, len(corpus)) # Setting top-k similarity scores to 5

for i, query in enumerate(docs):
    query_embedding = embedder.encode(query, convert_to_tensor=True) # Embedding the preprocessed session notes
    similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings) # Calculate cosine similarity score
    scores, indices = torch.topk(similarity_scores, k=top_k) # Selecting only the top-5 similarity scores

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        results.append({
            "Query Index": i,
            "Query Text": query,
            "Rank": rank,
            "Matched Sentence": corpus[idx],
            "Similarity Score": round(score.item(), 4)
        })

df_results = pd.DataFrame(results)
df_results.to_csv("semantic_search_results.csv", index=False) # Saving results
