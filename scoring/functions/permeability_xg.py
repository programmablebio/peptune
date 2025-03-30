import pandas as pd
import numpy as np
import optuna
from optuna.trial import TrialState
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from datasets import load_from_disk
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def save_and_plot_predictions(y_true_train, y_pred_train, y_true_val, y_pred_val, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Save training predictions
    train_df = pd.DataFrame({'True Permeability': y_true_train, 'Predicted Permeability': y_pred_train})
    train_df.to_csv(os.path.join(output_path, 'train_predictions.csv'), index=False)

    # Save validation predictions
    val_df = pd.DataFrame({'True Permeability': y_true_val, 'Predicted Permeability': y_pred_val})
    val_df.to_csv(os.path.join(output_path, 'val_predictions.csv'), index=False)

    # Plot training predictions
    plot_correlation(
        y_true_train,
        y_pred_train,
        title="Training Set Correlation Plot",
        output_file=os.path.join(output_path, 'train_correlation.png'),
    )

    # Plot validation predictions
    plot_correlation(
        y_true_val,
        y_pred_val,
        title="Validation Set Correlation Plot",
        output_file=os.path.join(output_path, 'val_correlation.png'),
    )

def plot_correlation(y_true, y_pred, title, output_file):
    spearman_corr, _ = spearmanr(y_true, y_pred)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Data points', color='#BC80FF')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='teal', linestyle='--', label='Ideal fit')

    # Add annotations
    plt.title(f"{title}\nSpearman Correlation: {spearman_corr:.3f}")
    plt.xlabel("True Permeability (logP)")
    plt.ylabel("Predicted Affinity (logP)")
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Load dataset
dataset = load_from_disk('/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/permeability/30K-data/')

# Extract sequences, labels, and embeddings
sequences = np.stack(dataset['sequence'])
labels = np.stack(dataset['labels'])  # Regression labels
embeddings = np.stack(dataset['embedding'])  # Pre-trained embeddings

# Function to compute Morgan fingerprints
def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            # If the SMILES string is invalid, use a zero vector
            fps.append(np.zeros(n_bits))
            print(f"Invalid SMILES: {smiles}")
    return np.array(fps)

# Compute Morgan fingerprints for the sequences
#morgan_fingerprints = compute_morgan_fingerprints(sequences)

# Concatenate embeddings with Morgan fingerprints
#input_features = np.concatenate([embeddings, morgan_fingerprints], axis=1)
input_features = embeddings

# Initialize global variables
best_model_path = "/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/permeability/30K-train"
os.makedirs(best_model_path, exist_ok=True)

def trial_info_callback(study, trial):
    if study.best_trial == trial:
        print(f"Trial {trial.number}:")
        print(f"  MSE: {trial.value}")

def objective(trial):
    # Define hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'lambda': trial.suggest_float('lambda', 0.1, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'tree_method': 'hist',
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10.0, log=True),
        'device': 'cuda:6',
    }
    """params = {
        'objective': 'reg:squarederror',
        'lambda': trial.suggest_float('lambda', 0.1, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2), 
        'max_depth': trial.suggest_int('max_depth', 4, 20),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), 
        'tree_method': 'hist',
        'device': 'cuda:6',
    }"""
    num_boost_round = trial.suggest_int('num_boost_round', 10, 1000)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(input_features, labels, test_size=0.2, random_state=42)

    # Convert data to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    # Train XGBoost
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predict and evaluate
    preds_train = model.predict(dtrain)
    preds_val = model.predict(dvalid)
    
    mse = mean_squared_error(y_val, preds_val)
    
    # Calculate Spearman Rank Correlation
    spearman_corr, _ = spearmanr(y_val, preds_val)
    print(f"Spearman Rank Correlation: {spearman_corr}")

    # Save the best model
    if trial.study.user_attrs.get("best_mse", np.inf) > mse:
        trial.study.set_user_attr("best_mse", mse)
        trial.study.set_user_attr("best_spearman", spearman_corr)  # Save the Spearman correlation
        model.save_model(os.path.join(best_model_path, "best_model.json"))
        save_and_plot_predictions(y_train, preds_train, y_val, preds_val, best_model_path)

    return mse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=200, callbacks=[trial_info_callback])

    # Print study statistics
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial value (MSE): {study.best_trial.value}")
    print(f"  Best Spearman Correlation: {study.user_attrs.get('best_spearman', None)}")  # Print the best Spearman correlation
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")