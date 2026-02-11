'''
K-Means Clustering Script

Here, we have already selected k=6 as the optimial K. 
K optimization was done using a separate script ("kmeans_optimization.py").

Input: Raw texts or preprocessed texts of counseling session notes
Output: clustered_queries_by_similarity.csv
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score, silhouette_samples

# Loading Preprocessed Counseling Session Notes
df = pd.read_csv("semantic_search/semantic_search_results.csv")
queries = df['Query Text'].drop_duplicates().tolist()

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

# Embed both query and corpus
embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_embeddings = embedder.encode(queries)
corpus_embeddings = embedder.encode(corpus)

# Re-compute cosine similarity vectors ("semantic_search.py" only saved the top-5 similarity scores)
similarity_matrix = cosine_similarity(query_embeddings, corpus_embeddings)

# Cluster based on similarity vectors
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(similarity_matrix)

# Compute overall silhouette score
sil_score = silhouette_score(similarity_matrix, cluster_labels)
print(f"Silhouette Score for k={n_clusters}: {sil_score:.3f}")

sample_silhouette_values = silhouette_samples(similarity_matrix, cluster_labels)
y_lower = 10

# Silhouette Plot
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_silhouette_vals = sample_silhouette_values[cluster_labels == i]
    cluster_silhouette_vals.sort()
    size = len(cluster_silhouette_vals)
    y_upper = y_lower + size

    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size, f"Cluster {i}")
    y_lower = y_upper + 10

plt.axvline(x=sil_score, color="red", linestyle="--")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.title("Silhouette Plot of Clusters Based on Similarity Vectors")
plt.tight_layout()
plt.show()

# Get most representative query (closest to cluster center)
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, similarity_matrix)
cluster_names = [queries[i] for i in closest]
cluster_to_label = {i: f"Cluster {i}: {cluster_names[i][:80]}..." for i in range(n_clusters)}
custom_labels = [cluster_to_label[c] for c in cluster_labels]

for i, text in enumerate(cluster_names):
    print(f"Cluster {i}: {text}")

# Save cluster assignments
cluster_df = pd.DataFrame({
    "Query Text": queries,
    "Cluster": cluster_labels
})
cluster_df.to_csv("clustered_queries_by_similarity.csv", index=False)

# Cluster visualization with t-SNE -------------------------------------
# Reduce similarity vectors using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(similarity_matrix)

plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=custom_labels, palette='tab10', s=70)
plt.title("t-SNE of Query Similarity to Corpus (Clustered)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Sort legend
handles, labels = scatter.get_legend_handles_labels()
sorted_pairs = sorted(zip(handles, labels), key=lambda x: int(x[1].split()[1].strip(":")))
sorted_handles, sorted_labels = zip(*sorted_pairs)

plt.legend(
    handles=sorted_handles,
    labels=sorted_labels,
    title="Cluster Themes",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.35),
    ncol=2,
    fontsize='small',
    title_fontsize='medium',
    frameon=False
)

# Label each cluster center on the plot
for i, idx in enumerate(closest):
    x, y = tsne_result[idx]
    plt.text(x, y, f"Cluster {i}", fontsize=10, weight='bold', color='black', ha='center', va='center')

plt.tight_layout()
