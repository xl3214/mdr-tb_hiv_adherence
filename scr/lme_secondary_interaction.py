"""
LME Interaction Analysis: Cluster × Time-Chunk Effects
Assesses whether cluster effects on adherence vary by time period

No cross-validation - fits full dataset to examine interaction patterns
Input: semantic_similarity_top5_BDQadherence.csv; semantic_similarity_top5_ARTadherence.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Configuration
med = "art" # "bdq"
output_dir = "lme_results/cluster_time_interaction"
os.makedirs(output_dir, exist_ok=True)

# Load and Prepare Data
df = pd.read_csv(f"semantic_similarity_top5_{med.upper()}adherence.csv")

# Create features
similarity_cols = [
    'top1_Similarity Score', 'top2_Similarity Score',
    'top3_Similarity Score', 'top4_Similarity Score',
    'top5_Similarity Score'
]

for i, col in enumerate(similarity_cols):
    df[f'sim_{i+1}'] = df[col]

df['adherence'] = df["Adherence"]

# LME Modeling: With Interactions between Chunk (categorical) and Cluster (categorical)
formula_interact = "adherence ~ sim_1 + sim_2 + sim_3 + sim_4 + sim_5 + C(Chunk) * C(Cluster)"
model_interact = smf.mixedlm(formula_interact, df, groups=df["PARTICIPANTID"])
fitted_interact = model_interact.fit(method='lbfgs', maxiter=100)

print("\nModel Summary:")
print(fitted_interact.summary())

# Extract Coefficients
interact_coefs = []
for param in fitted_interact.params.index:
    if 'C(Chunk)' in param and 'C(Cluster)' in param and ':' in param:
        interact_coefs.append({
            'Parameter': param,
            'Coefficient': fitted_interact.params[param],
            'Std_Error': fitted_interact.bse[param],
            'p_value': fitted_interact.pvalues[param],
            'CI_Lower': fitted_interact.conf_int().loc[param, 0],
            'CI_Upper': fitted_interact.conf_int().loc[param, 1]
        })

interact_df = pd.DataFrame(interact_coefs)
print(interact_df.to_string(index=False))

# Save Results
interact_coef_path = f"{output_dir}/{med}_interaction_coefficients.csv"
interact_df.to_csv(interact_coef_path, index=False, float_format='%.4f')

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nPreferred Model: {model_name}")
print(f"Output directory: {output_dir}/")
