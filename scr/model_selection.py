"""
Script: Model Comparison

Compares multiple models
1. Mean/Median Baseline (simplest possible)
2. Ridge Regression (linear model with regularization)
3. Random Forest (non-linear, tree-based)
4. Linear Mixed-Effects (statistically appropriate for nested data)
5. GAM (Generalized Additive Model - flexible non-linear relationships)
6. LSTM (initial approach)

All models use participant-level 5-fold CV for fair comparison

Input:  semantic_similarity_top5_{DRUG}adherence.csv
Output: model_comparison/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import statsmodels.formula.api as smf
from pygam import GAM, s, te, l
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(drug):
    """Load and preprocess data"""
    df = pd.read_csv(f"semantic_similarity_top5_{drug.upper()}adherence.csv")
    
    # Encode cluster
    ohe = OneHotEncoder(sparse_output=False)
    cluster_onehot = ohe.fit_transform(df[['Cluster']])
    
    # Encode chunk
    le_chunk = LabelEncoder()
    df["Chunk_Encoded"] = le_chunk.fit_transform(df["Chunk"])
    
    # Encode participant for random effects
    le_participant = LabelEncoder()
    df["Participant_Encoded"] = le_participant.fit_transform(df["PARTICIPANTID"])
    
    # Similarity features
    similarity_cols = [
        'top1_Similarity Score', 'top2_Similarity Score',
        'top3_Similarity Score', 'top4_Similarity Score',
        'top5_Similarity Score'
    ]
    
    # Create flat feature matrix (for non-LSTM models)
    # Match training order: [similarity, chunk, cluster]
    X_flat = np.concatenate([
        df[similarity_cols].values,
        df[["Chunk_Encoded"]].values,
        cluster_onehot
    ], axis=1).astype(np.float32)
    
    y = df["Adherence"].values.astype(np.float32)
    participants = df["PARTICIPANTID"].values
    participant_ids = df["Participant_Encoded"].values
    
    # Create dataframe for mixed-effects model
    df_lme = df.copy()
    for i, col in enumerate(similarity_cols):
        df_lme[f'sim_{i+1}'] = df[col]
    df_lme['chunk_encoded'] = df["Chunk_Encoded"]
    df_lme['adherence'] = y
    
    print(f"  Loaded {len(df)} notes from {len(np.unique(participants))} participants")
    print(f"  Features: {X_flat.shape[1]} (5 similarity + 1 chunk + {cluster_onehot.shape[1]} cluster)")
    
    return X_flat, y, participants, participant_ids, df_lme


def evaluate_baseline_models(X, y, participants):
    """Evaluate simple baseline models"""
    results = {}
    gkf = GroupKFold(n_splits=5)
    
    # Mean baseline
    print("\nMean Baseline (predict overall mean)")
    fold_maes = []
    fold_rmses = []
    for train_idx, test_idx in gkf.split(X, y, groups=participants):
        y_train, y_test = y[train_idx], y[test_idx]
        pred = np.full_like(y_test, y_train.mean())
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        fold_maes.append(mae)
        fold_rmses.append(rmse)
    
    results['Mean_Baseline'] = {
        'MAE_mean': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE_mean': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'Model_Type': 'Baseline'
    }
    
    # Median baseline
    print("\nMedian Baseline (predict overall median)")
    fold_maes = []
    fold_rmses = []
    for train_idx, test_idx in gkf.split(X, y, groups=participants):
        y_train, y_test = y[train_idx], y[test_idx]
        pred = np.full_like(y_test, np.median(y_train))
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        fold_maes.append(mae)
        fold_rmses.append(rmse)
    
    results['Median_Baseline'] = {
        'MAE_mean': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE_mean': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'Model_Type': 'Baseline'
    }
    
    return results


def evaluate_ridge(X, y, participants):
    """Ridge regression with participant-level CV"""
    gkf = GroupKFold(n_splits=5)
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=participants)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge with alpha=1.0 (default regularization)
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Constrain to [0,1]
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
      
    return {
        'Ridge_Regression': {
            'MAE_mean': np.mean(fold_maes),
            'MAE_std': np.std(fold_maes),
            'RMSE_mean': np.mean(fold_rmses),
            'RMSE_std': np.std(fold_rmses),
            'R2_mean': np.mean(fold_r2s),
            'R2_std': np.std(fold_r2s),
            'Model_Type': 'Linear'
        }
    }


def evaluate_random_forest(X, y, participants):
    """Random Forest with participant-level CV"""
    gkf = GroupKFold(n_splits=5)
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=participants)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train RF
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
            
    return {
        'Random_Forest': {
            'MAE_mean': np.mean(fold_maes),
            'MAE_std': np.std(fold_maes),
            'RMSE_mean': np.mean(fold_rmses),
            'RMSE_std': np.std(fold_rmses),
            'R2_mean': np.mean(fold_r2s),
            'R2_std': np.std(fold_r2s),
            'Model_Type': 'Tree-based'
        }
    }


def evaluate_mixed_effects(df_lme, participants):
    """
    Linear Mixed-Effects Model with participant random intercepts
    """  
    try:
        gkf = GroupKFold(n_splits=5)
        
        fold_maes = []
        fold_rmses = []
        fold_r2s = []
        
        # Build formula: fixed effects for all features + random intercept per participant
        formula = "adherence ~ sim_1 + sim_2 + sim_3 + sim_4 + sim_5 + chunk_encoded + C(Cluster)"
        
        print(f"Formula: {formula} + (1|PARTICIPANTID)")
        
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df_lme, df_lme['adherence'], groups=participants)):
            # Split data
            df_train = df_lme.iloc[train_idx].copy()
            df_test = df_lme.iloc[test_idx].copy()
            
            # Fit model on training data
            try:
                model = smf.mixedlm(formula, df_train, groups=df_train["PARTICIPANTID"]).fit(method='powell', maxiter=500)
            except:
                # If powell fails, try lbfgs
                model = smf.mixedlm(formula, df_train, groups=df_train["PARTICIPANTID"]).fit(method='lbfgs', maxiter=500)
            
            # Predict on entire test set at once
            # For participants not in training: prediction uses only fixed effects (random effect = 0)
            # For participants in training: uses their estimated random effect
            y_pred = model.predict(df_test)
            y_pred = np.clip(y_pred.values, 0, 1)
            y_test = df_test['adherence'].values
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)
        
        return {
            'Linear_Mixed_Effects': {
                'MAE_mean': np.mean(fold_maes),
                'MAE_std': np.std(fold_maes),
                'RMSE_mean': np.mean(fold_rmses),
                'RMSE_std': np.std(fold_rmses),
                'R2_mean': np.mean(fold_r2s),
                'R2_std': np.std(fold_r2s),
                'Model_Type': 'Mixed-Effects'
            }
        }
    
    except Exception as e:
        print(f"\nError fitting mixed-effects model: {e}")
        import traceback
        traceback.print_exc()
        return {}
    

def evaluate_gam(X, y, participants):
    """
    Generalized Additive Model
    Captures non-linear relationships through smooth functions
    (No random effects - separate from LME approach)
    """
    try:
        gkf = GroupKFold(n_splits=5)
        
        fold_maes = []
        fold_rmses = []
        fold_r2s = []
        
        # X columns: [sim1, sim2, sim3, sim4, sim5, chunk, cluster_onehot...]
        n_similarity = 5
        chunk_idx = 5
        cluster_start = 6
        n_features = X.shape[1]
        
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=participants)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Build GAM formula
            # s() for smooth splines on continuous features
            # l() for linear terms on categorical features (cluster)
            
            gam_formula = (
                s(0, n_splines=5, lam=0.6) +  # sim1 - smooth with moderate regularization
                s(1, n_splines=5, lam=0.6) +  # sim2
                s(2, n_splines=5, lam=0.6) +  # sim3
                s(3, n_splines=5, lam=0.6) +  # sim4
                s(4, n_splines=5, lam=0.6) +  # sim5
                s(5, n_splines=4, lam=0.6)    # chunk (ordinal but allow smooth)
            )
            
            # Add linear terms for cluster (one-hot encoded categories)
            for i in range(cluster_start, n_features):
                gam_formula += l(i, lam=0.1)
            
            # Fit GAM
            model = GAM(gam_formula, fit_intercept=True)
            model.gridsearch(X_train, y_train, progress=False)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, 1)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)
        
        return {
            'GAM': {
                'MAE_mean': np.mean(fold_maes),
                'MAE_std': np.std(fold_maes),
                'RMSE_mean': np.mean(fold_rmses),
                'RMSE_std': np.std(fold_rmses),
                'R2_mean': np.mean(fold_r2s),
                'R2_std': np.std(fold_r2s),
                'Model_Type': 'Additive'
            }
        }
    
    except Exception as e:
        print(f"\nError fitting GAM: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
def evaluate_lstm(drug, X, y, participants):
    """Evaluate existing trained LSTM model"""
    model_path = f"LSTM/best_lstm_model_{drug}_participantSplit.keras"
    
    try:
        model = load_model(model_path, compile=False)
        print(f"Loaded LSTM model: {model_path}")
        model.summary()
    except Exception as e:
        print(f"LSTM model not found: {model_path}")
        print(f"Error: {e}")
        print("\nSkipping LSTM evaluation. Train model first with:")
        print(f"  python lstm_training_script.py")
        return {}
    
    # Reshape for LSTM (samples, timesteps=1, features)
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    
    gkf = GroupKFold(n_splits=5)
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=participants)):
        X_test = X_lstm[test_idx]
        y_test = y[test_idx]
        
        # Predict
        y_pred = model.predict(X_test, verbose=0).flatten()
        y_pred = np.clip(y_pred, 0, 1)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

    return {
        'LSTM': {
            'MAE_mean': np.mean(fold_maes),
            'MAE_std': np.std(fold_maes),
            'RMSE_mean': np.mean(fold_rmses),
            'RMSE_std': np.std(fold_rmses),
            'R2_mean': np.mean(fold_r2s),
            'R2_std': np.std(fold_r2s),
            'Model_Type': 'Deep Learning'
        }
    }


def compare_models(drug):
    """Run all model comparisons for one drug"""   
    # Load data
    X, y, participants, participant_ids, df_lme = load_and_prepare_data(drug)
    
    # Run all evaluations
    all_results = {}
    
    all_results.update(evaluate_baseline_models(X, y, participants))
    all_results.update(evaluate_ridge(X, y, participants))
    all_results.update(evaluate_random_forest(X, y, participants))
    all_results.update(evaluate_mixed_effects(df_lme, participants))
    all_results.update(evaluate_gam(X, y, participants))  # Changed from evaluate_gamm
    all_results.update(evaluate_lstm(drug, X, y, participants))
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results).T
    comparison_df.insert(0, 'Model', comparison_df.index)
    comparison_df = comparison_df.reset_index(drop=True)
    
    # Sort by MAE
    comparison_df = comparison_df.sort_values('MAE_mean')
    
    return comparison_df

def main():
    """Compare models for both drugs"""
    # Create output directory
    output_dir = "model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    all_comparisons = []
    
    for drug in ["art", "bdq"]:
        comparison = compare_models(drug)
        comparison.insert(0, 'Drug', drug.upper())
        all_comparisons.append(comparison)

        # Print summary
        display_cols = ['Model', 'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std', 'Model_Type']
        print(comparison[display_cols].to_string(index=False))
    
    # Combined table
    combined = pd.concat(all_comparisons, ignore_index=True)
    combined_path = f"{output_dir}/all_models_comparison.csv"
    combined.to_csv(combined_path, index=False, float_format='%.4f')
    

if __name__ == "__main__":
    main()
