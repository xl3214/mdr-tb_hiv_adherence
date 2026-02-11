'''
BERTopic Script for Topic Modeling

Input: Raw texts or preprocessed texts of counseling session notes
Output: BERTopic Model and Wordclouds
'''

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# Loading Counseling Session Notes with Preprocessing Function (from "preprocessing.py")
docs = load_and_preprocess("/Users/kristal99/Desktop/DA/PRAXIS/NLP/Rproj/data/mci1_preprocess.xlsx")
# Due to participant identification purposes, the raw texts will not be made available.
# For replication purposes, preprocessed texts are made available ("preprocessed_query_texts.csv")
# The analysis results can be replicated without going through the `load_and_preprocess` function. 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # SBERT Embedding Model

# umap_model = UMAP(random_state=42) # Our run did not set seed for randomization (result optimization) purposes
hdbscan_model = HDBSCAN(min_cluster_size=10, prediction_data=True)

# Run BERTopic Modeling, with n_gram tuning
topic_model = BERTopic(min_topic_size=3, n_gram_range=(1, 10), embedding_model=embedding_model, verbose=True, 
                       hdbscan_model=hdbscan_model, calculate_probabilities=True)

topics, probs = topic_model.fit_transform(docs)
topic_info = topic_model.get_topic_info().query("Topic != -1").sort_values(by='Count', ascending=False)
top_10 = topic_info.head(10).reset_index(drop=True)
topics_dict = topic_model.get_topics()

# Plot word clouds
for i in range(0, len(top_10), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for j in range(2):
        idx = i + j
        if idx < len(top_10):
            topic_id = top_10.loc[idx, 'Topic']
            count = top_10.loc[idx, 'Count']
            word_freq = dict(topics_dict[topic_id])
            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[j].imshow(wc, interpolation='bilinear')
            axes[j].axis('off')
            axes[j].set_title(f"Topic {topic_id} (Count: {count})")
    plt.tight_layout()
    plt.savefig(f"topic_wordcloud_{i}.png", dpi=300)

# Saving the model
topic_model.save("/Users/kristal99/Desktop/DA/PRAXIS/NLP")
