"""
Script: Clustering Optimization Analysis

Systematically evaluates clustering quality for k=2 to k=15:
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score
- Within-cluster sum of squares (elbow method)
- Topic coherence (C_V)

Outputs supplemental table comparing cluster quality metrics

Input:  semantic_search_results.csv
Output: (comparison tables and plots)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def compute_clustering_metrics(similarity_matrix, k_range):
    """
    Compute multiple clustering quality metrics for different k values
    """
    print("\n" + "="*80)
    print("CLUSTERING OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"\nEvaluating k from {k_range[0]} to {k_range[-1]}")
    
    results = []
    
    for k in k_range:
        print(f"\nEvaluating k={k}...")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(similarity_matrix)
        
        # Silhouette score (higher is better, range [-1, 1])
        sil_score = silhouette_score(similarity_matrix, labels)
        
        # Davies-Bouldin index (lower is better, minimum 0)
        db_score = davies_bouldin_score(similarity_matrix, labels)
        
        # Calinski-Harabasz score (higher is better)
        ch_score = calinski_harabasz_score(similarity_matrix, labels)
        
        # Within-cluster sum of squares (for elbow method)
        wcss = kmeans.inertia_
        
        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        min_cluster_size = counts.min()
        max_cluster_size = counts.max()
        cluster_size_std = counts.std()
        
        results.append({
            'k': k,
            'silhouette_score': sil_score,
            'davies_bouldin_index': db_score,
            'calinski_harabasz_score': ch_score,
            'wcss': wcss,
            'min_cluster_size': min_cluster_size,
            'max_cluster_size': max_cluster_size,
            'cluster_size_std': cluster_size_std
        })
        
        print(f"  Silhouette: {sil_score:.4f}")
        print(f"  Davies-Bouldin: {db_score:.4f}")
        print(f"  Calinski-Harabasz: {ch_score:.2f}")
        print(f"  Cluster sizes: {min_cluster_size}-{max_cluster_size} (std={cluster_size_std:.1f})")
    
    return pd.DataFrame(results)


def compute_coherence_scores(queries, similarity_matrix, k_range):
    """
    Compute topic coherence (C_V) for different k values
    """
    print("\n" + "-"*80)
    print("Computing Topic Coherence Scores")
    print("-"*80)
    
    coherence_results = []
    
    for k in k_range:
        print(f"\nComputing coherence for k={k}...")
        
        # Cluster
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(similarity_matrix)
        
        # Preprocess queries
        preprocessed_queries = [
            [word for word in simple_preprocess(q) if word not in ENGLISH_STOP_WORDS]
            for q in queries
        ]
        
        # Extract cluster keywords
        cluster_keywords = []
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_tokens = [preprocessed_queries[i] for i in cluster_indices]
            # Flatten and deduplicate
            flat = list(set(word for sublist in cluster_tokens for word in sublist))
            cluster_keywords.append(flat)
        
        # Compute coherence
        try:
            dictionary = Dictionary(preprocessed_queries)
            coherence_model = CoherenceModel(
                topics=cluster_keywords,
                texts=preprocessed_queries,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
        except:
            coherence = np.nan
        
        coherence_results.append({
            'k': k,
            'coherence_cv': coherence
        })
        
        print(f"  Coherence (C_V): {coherence:.4f}")
    
    return pd.DataFrame(coherence_results)


def find_optimal_k(metrics_df, coherence_df):
    """
    Suggest optimal k based on multiple criteria
    """
    print("\n" + "="*80)
    print("OPTIMAL K SELECTION")
    print("="*80)
    
    # Merge metrics
    full_metrics = metrics_df.merge(coherence_df, on='k')
    
    # Normalize metrics (0-1 scale) for comparison
    # Higher is better: silhouette, CH, coherence
    # Lower is better: DB, WCSS
    
    full_metrics['sil_norm'] = (full_metrics['silhouette_score'] - full_metrics['silhouette_score'].min()) / \
                                (full_metrics['silhouette_score'].max() - full_metrics['silhouette_score'].min())
    
    full_metrics['db_norm'] = 1 - (full_metrics['davies_bouldin_index'] - full_metrics['davies_bouldin_index'].min()) / \
                               (full_metrics['davies_bouldin_index'].max() - full_metrics['davies_bouldin_index'].min())
    
    full_metrics['ch_norm'] = (full_metrics['calinski_harabasz_score'] - full_metrics['calinski_harabasz_score'].min()) / \
                               (full_metrics['calinski_harabasz_score'].max() - full_metrics['calinski_harabasz_score'].min())
    
    full_metrics['coh_norm'] = (full_metrics['coherence_cv'] - full_metrics['coherence_cv'].min()) / \
                                (full_metrics['coherence_cv'].max() - full_metrics['coherence_cv'].min())
    
    # Composite score (equal weights)
    full_metrics['composite_score'] = (
        full_metrics['sil_norm'] + 
        full_metrics['db_norm'] + 
        full_metrics['ch_norm'] + 
        full_metrics['coh_norm']
    ) / 4
    
    # Find optimal
    optimal_idx = full_metrics['composite_score'].idxmax()
    optimal_k = full_metrics.loc[optimal_idx, 'k']
    
    print(f"\nRecommended optimal k: {optimal_k}")
    print("\nTop 3 candidates:")
    top3 = full_metrics.nlargest(3, 'composite_score')[['k', 'composite_score', 
                                                          'silhouette_score', 
                                                          'davies_bouldin_index',
                                                          'coherence_cv']]
    print(top3.to_string(index=False))
    
    print("\n" + "-"*80)
    print("Interpretation:")
    print(f"  Current k=6: Composite score = {full_metrics[full_metrics['k']==6]['composite_score'].values[0]:.4f}")
    print(f"  Optimal k={optimal_k}: Composite score = {full_metrics.loc[optimal_idx, 'composite_score']:.4f}")
    
    return full_metrics


def create_optimization_plots(metrics_df, coherence_df, output_dir):
    """
    Create visualization plots for clustering optimization
    """
    print("\n" + "-"*80)
    print("Creating Visualization Plots")
    print("-"*80)
    
    ensure_dir(output_dir)
    
    # Merge data
    full_metrics = metrics_df.merge(coherence_df, on='k')
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Silhouette Score
    axes[0, 0].plot(full_metrics['k'], full_metrics['silhouette_score'], 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 0].axvline(x=6, color='red', linestyle='--', alpha=0.5, label='k=6 (current)')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score (Higher is Better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Davies-Bouldin Index
    axes[0, 1].plot(full_metrics['k'], full_metrics['davies_bouldin_index'], 
                    marker='s', linewidth=2, markersize=8, color='orange')
    axes[0, 1].axvline(x=6, color='red', linestyle='--', alpha=0.5, label='k=6 (current)')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Davies-Bouldin Index')
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Calinski-Harabasz Score
    axes[1, 0].plot(full_metrics['k'], full_metrics['calinski_harabasz_score'], 
                    marker='^', linewidth=2, markersize=8, color='green')
    axes[1, 0].axvline(x=6, color='red', linestyle='--', alpha=0.5, label='k=6 (current)')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is Better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Topic Coherence
    axes[1, 1].plot(full_metrics['k'], full_metrics['coherence_cv'], 
                    marker='D', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axvline(x=6, color='red', linestyle='--', alpha=0.5, label='k=6 (current)')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Coherence (C_V)')
    axes[1, 1].set_title('Topic Coherence (Higher is Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "clustering_optimization_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {plot_path}")
    
    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['k'], metrics_df['wcss'], marker='o', linewidth=2, markersize=8)
    plt.axvline(x=6, color='red', linestyle='--', alpha=0.5, label='k=6 (current)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method for Optimal k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = Path(output_dir) / "elbow_plot.png"
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {elbow_path}")


def main():
    """Run clustering optimization analysis"""
    print("="*80)
    print("CLUSTERING OPTIMIZATION")
    print("="*80)
    
    # Setup paths
    input_file = "semantic_search_results.csv"
    output_dir = Path("clustering_optimization")
    ensure_dir(output_dir)
    
    # Reference corpus
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
        "The session ended with affirming patient's efforts and planning next steps."
    ]
    
    # Load data
    if not Path(input_file).exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Please ensure the semantic search results file exists."
        )
    
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    queries = df['Query Text'].drop_duplicates().tolist()
    print(f"Loaded {len(queries)} unique queries")
    print(f"Number of themes in corpus: {len(corpus)}")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = embedder.encode(queries, show_progress_bar=True)
    corpus_embeddings = embedder.encode(corpus, show_progress_bar=True)
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Evaluate k from 2 to 15
    k_range = range(2, 16)
    
    # Compute clustering metrics
    metrics_df = compute_clustering_metrics(similarity_matrix, k_range)
    
    # Compute coherence scores
    coherence_df = compute_coherence_scores(queries, similarity_matrix, k_range)
    
    # Find optimal k
    full_metrics = find_optimal_k(metrics_df, coherence_df)
    
    # Create plots
    create_optimization_plots(metrics_df, coherence_df, output_dir)
    
    # Save results
    print("\n" + "-"*80)
    print("Saving Results")
    print("-"*80)
    
    # Full table for supplemental materials
    supplement_path = output_dir / "supplemental_table_clustering_evaluation.csv"
    full_metrics[['k', 'silhouette_score', 'davies_bouldin_index', 
                  'calinski_harabasz_score', 'coherence_cv', 'composite_score',
                  'min_cluster_size', 'max_cluster_size', 'cluster_size_std']].to_csv(
        supplement_path, index=False, float_format='%.4f'
    )
    print(f"  Saved supplemental table: {supplement_path}")
    
    # Summary for main text
    summary_path = output_dir / "clustering_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CLUSTERING OPTIMIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Evaluated k from 2 to 15\n\n")
        f.write("Current choice (k=6) metrics:\n")
        k6_metrics = full_metrics[full_metrics['k']==6].iloc[0]
        f.write(f"  Silhouette Score: {k6_metrics['silhouette_score']:.4f}\n")
        f.write(f"  Davies-Bouldin Index: {k6_metrics['davies_bouldin_index']:.4f}\n")
        f.write(f"  Calinski-Harabasz Score: {k6_metrics['calinski_harabasz_score']:.2f}\n")
        f.write(f"  Topic Coherence (C_V): {k6_metrics['coherence_cv']:.4f}\n")
        f.write(f"  Composite Score: {k6_metrics['composite_score']:.4f}\n\n")
        
        optimal_k = full_metrics.loc[full_metrics['composite_score'].idxmax(), 'k']
        f.write(f"Statistically optimal k: {optimal_k}\n\n")
        f.write("Top 3 candidates:\n")
        top3 = full_metrics.nlargest(3, 'composite_score')[['k', 'composite_score', 
                                                              'silhouette_score', 
                                                              'davies_bouldin_index',
                                                              'coherence_cv']]
        f.write(top3.to_string(index=False))

    print(f"  Saved summary: {summary_path}")
    
    print("\n" + "="*80)
    print("CLUSTERING OPTIMIZATION COMPLETE")
    print("="*80)
    optimal_k = full_metrics.loc[full_metrics['composite_score'].idxmax(), 'k']
    print(f"Optimal k: {optimal_k}")
    print(f"k=6 composite score: {full_metrics[full_metrics['k']==6]['composite_score'].values[0]:.4f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"1. Supplemental table: {supplement_path.name}")
    print(f"2. Visualization plots: clustering_optimization_metrics.png, elbow_plot.png")
    print(f"3. Text summary: {summary_path.name}")


if __name__ == '__main__':
    main()
