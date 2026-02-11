"""
Linear Mixed-Effects Model for Adherence Prediction
This script fits a linear mixed-effects model to predict adherence scores
based on semantic similarity features, chunk, and cluster, while accounting
for participant-level random effects. The model is evaluated using 5-fold
participant-level cross-validation.

Predictors: Top 5 semantic similarity scores, chunk (encoded), cluster (categorical)
Outcome: Adherence (continuous, 0-1)
Random Effects: Uses participant random intercepts to account for individual heterogeneity
Evaluation: 5-fold participant-level cross-validation

Input: semantic_similarity_top5_BDQadherence.csv; semantic_similarity_top5_ARTadherence.csv
Output: LME modeling results
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.formula.api as smf
import warnings
import pickle
warnings.filterwarnings('ignore')

# Configuration
med = "art"  # "bdq"
output_dir = "lme_results"
os.makedirs(output_dir, exist_ok=True)

# Load and Prepare Data
df = pd.read_csv(f"semantic_similarity_top5_{med.upper()}adherence.csv")

# Create feature columns for the model formula
similarity_cols = [
    'top1_Similarity Score', 'top2_Similarity Score',
    'top3_Similarity Score', 'top4_Similarity Score',
    'top5_Similarity Score'
]

# Rename for easier formula use
for i, col in enumerate(similarity_cols):
    df[f'sim_{i+1}'] = df[col]

# Encode chunk as numeric
le_chunk = LabelEncoder()
df['chunk_encoded'] = le_chunk.fit_transform(df["Chunk"])

# Target variable
df['adherence'] = df["Adherence"]

# 5-Fold Cross-Validation
participants = df["PARTICIPANTID"].unique()
gkf = GroupKFold(n_splits=5)

fold_results = []
all_predictions = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['PARTICIPANTID'])):
    
    # Split by participants
    df_train = df.iloc[train_idx].copy()
    train_participants = df_train['PARTICIPANTID'].unique()
    df_test = df.iloc[test_idx].copy()
    test_participants = df_test['PARTICIPANTID'].unique()
    
    print(f"Training: {len(train_participants)} participants, {len(df_train)} notes")
    print(f"Testing:  {len(test_participants)} participants, {len(df_test)} notes")
    
    # Build formula: fixed effects + random intercept per participant
    formula = "adherence ~ sim_1 + sim_2 + sim_3 + sim_4 + sim_5 + C(chunk_encoded) + C(Cluster)"
    
    # Fit model on training data
    model = smf.mixedlm(formula, df_train, groups=df_train["PARTICIPANTID"])
    fitted_model = model.fit(method='lbfgs', maxiter=100)
    
    # Make predictions on test data
    # For new participants, this uses fixed effects only (population average)
    y_pred = fitted_model.predict(df_test)
    y_true = df_test['adherence'].values
    
    # Clip predictions to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    fold_results.append({
        'fold': fold_idx + 1,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_train_participants': len(train_participants),
        'n_test_participants': len(test_participants),
        'n_train_samples': len(df_train),
        'n_test_samples': len(df_test)
    })
    
    # Store predictions for this fold
    fold_predictions = df_test[['PARTICIPANTID', 'VISITCODE', 'Chunk', 'adherence']].copy()
    fold_predictions['predicted_adherence'] = y_pred
    fold_predictions['fold'] = fold_idx + 1
    all_predictions.append(fold_predictions)

# Summary Statistics
from scipy import stats 
results_df = pd.DataFrame(fold_results)

# Calculate 95% CI for all metrics
n_folds = len(results_df)
t_crit = stats.t.ppf(0.975, df=n_folds-1)

# MAE
mae_mean = results_df['mae'].mean()
mae_std = results_df['mae'].std()
mae_se = mae_std / np.sqrt(n_folds)
mae_ci_lower = mae_mean - (t_crit * mae_se)
mae_ci_upper = mae_mean + (t_crit * mae_se)

# RMSE
rmse_mean = results_df['rmse'].mean()
rmse_std = results_df['rmse'].std()
rmse_se = rmse_std / np.sqrt(n_folds)
rmse_ci_lower = rmse_mean - (t_crit * rmse_se)
rmse_ci_upper = rmse_mean + (t_crit * rmse_se)

# R²
r2_mean = results_df['r2'].mean()
r2_std = results_df['r2'].std()
r2_se = r2_std / np.sqrt(n_folds)
r2_ci_lower = r2_mean - (t_crit * r2_se)
r2_ci_upper = r2_mean + (t_crit * r2_se)

# Fit Final Model on All Data

formula = "adherence ~ sim_1 + sim_2 + sim_3 + sim_4 + sim_5 + C(chunk_encoded) + C(Cluster)"
final_model = smf.mixedlm(formula, df, groups=df["PARTICIPANTID"])
final_fitted = final_model.fit(method='lbfgs', maxiter=100)

# Get fitted values
df['predicted_adherence'] = final_fitted.fittedvalues
df['residual'] = df['adherence'] - df['predicted_adherence']
df['abs_error'] = np.abs(df['residual'])

# Overall metrics on training data
overall_mae = df['abs_error'].mean()
overall_rmse = np.sqrt(mean_squared_error(df['adherence'], df['predicted_adherence']))
overall_r2 = r2_score(df['adherence'], df['predicted_adherence'])

# =========================
# Save Results
# =========================
# Saving model 
model_path = f"{output_dir}/{med}_lme_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_fitted, f)

# CV fold results
cv_results_path = f"{output_dir}/{med}_lme_cv_results.csv"
results_df.to_csv(cv_results_path, index=False, float_format='%.4f')

# All predictions from CV
all_predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_path = f"{output_dir}/{med}_lme_cv_predictions.csv"
all_predictions_df.to_csv(predictions_path, index=False, float_format='%.4f')

# Full dataset predictions
full_predictions_path = f"{output_dir}/{med}_lme_full_predictions.csv"
df[['PARTICIPANTID', 'VISITCODE', 'Chunk', 'adherence', 'predicted_adherence', 
    'residual', 'abs_error']].to_csv(full_predictions_path, index=False, float_format='%.4f')

# Model coefficients
coef_summary = pd.DataFrame({
    'Parameter': final_fitted.params.index,
    'Coefficient': final_fitted.params.values,
    'Std_Error': final_fitted.bse.values,
    'p_value': final_fitted.pvalues.values
})
coef_path = f"{output_dir}/{med}_lme_coefficients.csv"
coef_summary.to_csv(coef_path, index=False, float_format='%.4f')
