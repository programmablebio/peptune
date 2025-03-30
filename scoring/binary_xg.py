import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
import optuna
from optuna.trial import TrialState
import xgboost as xgb
import os
from datasets import load_from_disk
from lightning.pytorch import seed_everything
from rdkit import Chem, rdBase, DataStructs
from typing import List
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns

def save_and_plot_binary_predictions(y_true_train, y_pred_train, y_true_val, y_pred_val, threshold, output_path):
    """
    Saves the true and predicted values for training and validation sets, and generates binary classification plots.

    Parameters:
        y_true_train (array): True labels for the training set.
        y_pred_train (array): Predicted probabilities for the training set.
        y_true_val (array): True labels for the validation set.
        y_pred_val (array): Predicted probabilities for the validation set.
        threshold (float): Classification threshold for predictions.
        output_path (str): Directory to save the CSV files and plots.
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert probabilities to binary predictions
    y_pred_train_binary = (y_pred_train >= threshold).astype(int)
    y_pred_val_binary = (y_pred_val >= threshold).astype(int)

    # Save training predictions
    train_df = pd.DataFrame({
        'True Label': y_true_train,
        'Predicted Probability': y_pred_train,
        'Predicted Label': y_pred_train_binary
    })
    train_df.to_csv(os.path.join(output_path, 'train_predictions_binary.csv'), index=False)

    # Save validation predictions
    val_df = pd.DataFrame({
        'True Label': y_true_val,
        'Predicted Probability': y_pred_val,
        'Predicted Label': y_pred_val_binary
    })
    val_df.to_csv(os.path.join(output_path, 'val_predictions_binary.csv'), index=False)

    # Plot training predictions
    plot_boxplot_with_threshold(
        y_true_train,
        y_pred_train,
        threshold,
        title="Training Set Binary Classification Plot",
        output_file=os.path.join(output_path, 'train_classification_plot.png')
    )

    # Plot validation predictions
    plot_boxplot_with_threshold(
        y_true_val,
        y_pred_val,
        threshold,
        title="Validation Set Binary Classification Plot",
        output_file=os.path.join(output_path, 'val_classification_plot.png')
    )

def plot_binary_correlation(y_true, y_pred, threshold, title, output_file):
    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Data points', color='#BC80FF')

    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')

    # Add annotations
    plt.title(title)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Probability")
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_boxplot_with_threshold(y_true, y_pred, threshold, title, output_file):
    """
    Generates a boxplot for binary classification and includes a threshold line.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted probabilities.
        threshold (float): Classification threshold for predictions.
        title (str): Title of the plot.
        output_file (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))

    # Combine data into a DataFrame for seaborn
    df = pd.DataFrame({'True Label': y_true, 'Predicted Probability': y_pred})

    # Boxplot
    sns.boxplot(x='True Label', y='Predicted Probability', data=df)

    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.text(
        x=0.5, y=threshold + 0.05, s=f"Threshold = {threshold}", color="red", fontsize=10
    )

    # Add annotations
    plt.title(title)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_boxplot(y_true, y_pred, title, output_file):
    plt.figure(figsize=(10, 8))

    # Combine data into a single DataFrame for seaborn
    df = pd.DataFrame({'True Label': y_true, 'Predicted Probability': y_pred})
    sns.boxplot(x='True Label', y='Predicted Probability', data=df)

    # Add annotations
    plt.title(title)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    
def plot_binary_correlation_with_density(y_true, y_pred, threshold, title, output_file):
    """
    Generates a scatter plot with a density plot for binary classification and saves it to a file.
    """
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(range(len(y_true)), y_pred, alpha=0.5, label='Predicted Probabilities', color='#BC80FF')

    # Add density plot
    sns.kdeplot(y_pred, color='green', fill=True, alpha=0.3, label='Probability Density')

    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')

    # Add annotations
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Predicted Probability")
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

seed_everything(42)

dataset = load_from_disk('/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/solubility/new_data') 

sequences = np.stack(dataset['sequence'])  # Ensure sequences are SMILES strings
labels = np.stack(dataset['labels']) 
embeddings = np.stack(dataset['embedding'])

# Initialize best F1 score and model path
best_f1 = -np.inf
best_model_path = "/home/st512/peptune/scripts/peptide-mdlm-mcts/scoring/functions/solubility/new_train/"

# Trial callback
def trial_info_callback(study, trial):
    if study.best_trial == trial:
        print(f"Trial {trial.number}:")
        print(f"  Weighted F1 Score: {trial.value}")
        

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'tree_method': 'hist',
        'device': 'cuda:0',
    }
    num_boost_round = trial.suggest_int('num_boost_round', 10, 1000)

    # Split the data
    train_idx, val_idx = train_test_split(
        np.arange(len(sequences)), test_size=0.2, stratify=labels, random_state=42
    )
    train_subset = dataset.select(train_idx).with_format("torch")
    val_subset = dataset.select(val_idx).with_format("torch")

    # Extract embeddings and labels for train/validation
    train_embeddings = train_subset['embedding']
    valid_embeddings = val_subset['embedding']
    train_labels = train_subset['labels']
    valid_labels = val_subset['labels']

    # Prepare training and validation sets
    dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    dvalid = xgb.DMatrix(valid_embeddings, label=valid_labels)

    # Train the model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predict probabilities
    preds_train = model.predict(dtrain)
    preds_val = model.predict(dvalid)

    # Perform dynamic thresholding on validation predictions
    best_f1_val = -np.inf
    best_threshold = 0.5

    for threshold in np.arange(0.1, 1.0, 0.05):  # Try thresholds from 0.1 to 1.0
        preds_val_binary = (preds_val >= threshold).astype(int)
        f1_temp = f1_score(valid_labels, preds_val_binary, average="weighted")
        if f1_temp > best_f1_val:
            best_f1_val = f1_temp
            best_threshold = threshold

    print(f"Best F1 Score: {best_f1_val:.3f} at Threshold: {best_threshold:.3f}")

    # Calculate AUC for additional insight
    auc_val = roc_auc_score(valid_labels, preds_val)
    print(f"AUC: {auc_val:.3f}")

    # Save the best model if the F1 score is improved
    if trial.study.user_attrs.get("best_f1", -np.inf) < best_f1_val:
        trial.study.set_user_attr("best_f1", best_f1_val)
        trial.study.set_user_attr("best_threshold", best_threshold)  # Save the best threshold
        os.makedirs(best_model_path, exist_ok=True)

        model.save_model(os.path.join(best_model_path, "best_model.json"))
        print(f"Best model saved to {os.path.join(best_model_path, 'best_model.json')}")

        # Save and plot binary predictions with the best threshold
        save_and_plot_binary_predictions(
            train_labels,
            preds_train,
            valid_labels,
            preds_val,
            best_threshold,
            best_model_path
        )

    return best_f1_val

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=200)
        
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best AUC: {study.user_attrs.get('best_auc', None)}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")