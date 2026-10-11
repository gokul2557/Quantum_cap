"""
train_models.py
---------------
End-to-end training and evaluation script for the Quantum AI Fraud Detection prototype.
Trains both Classical Random Forest Baseline and Quantum VQC Classifier,
evaluates on test data, and exports model artifacts and comparison reports.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from generate_dataset import save_dataset
from preprocessing.preprocessing import load_and_clean_data, prepare_data, get_quantum_training_subset
from models.classical_model import ClassicalFraudModel
from models.quantum_model import QuantumFraudVQC
from evaluation.evaluate import (
    compute_model_metrics,
    generate_comparison_table,
    plot_side_by_side_confusion_matrices,
    plot_roc_curves
)

def train_and_evaluate(
    data_path: str = "data/transactions.csv",
    quantum_target_samples: int = 1000,
    fraud_weight: float = 0.70,
    cobyla_max_iter: int = 40,
    random_seed: int = 42
):
    print("=" * 70)
    print("   QUANTUM AI REAL-TIME FRAUD DETECTION: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Dataset Generation / Loading
    if not os.path.exists(data_path):
        print(f"[Pipeline] Data file '{data_path}' not found. Generating synthetic dataset...")
        save_dataset(data_path, n_samples=5000)
    
    df = load_and_clean_data(data_path)
    
    # 2. Preprocessing & Stratified Train/Test Split (Strictly 80/20, no data leakage)
    print("\n--- Phase 3: Preprocessing & Scaling (No Leakage) ---")
    data = prepare_data(df, test_size=0.2, random_state=random_seed)
    X_train_full = data["X_train_full"]
    X_test_full = data["X_test_full"]
    X_train_q = data["X_train_q"]
    X_test_q = data["X_test_q"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"Total Training Pool : {len(y_train)} (Legitimate: {np.sum(y_train==0)}, Fraud: {np.sum(y_train==1)})")
    print(f"Unseen Test Samples : {len(y_test)} (Legitimate: {np.sum(y_test==0)}, Fraud: {np.sum(y_test==1)})")

    # 3. Train Classical Baseline
    print("\n--- Phase 4: Training Classical Baseline (Random Forest) ---")
    start_time = time.time()
    classical_model = ClassicalFraudModel(random_state=random_seed)
    classical_model.train(X_train_full, y_train)
    rf_train_time = time.time() - start_time
    print(f"[Classical] Training completed in {rf_train_time:.2f} seconds.")

    # Evaluate Classical Model on ALL 1,000 test samples
    rf_pred = classical_model.predict(X_test_full)
    rf_prob = classical_model.predict_proba(X_test_full)[:, 1]
    rf_metrics = compute_model_metrics("Random Forest (Classical)", y_test, rf_pred, rf_prob)
    classical_model.save("models/random_forest.joblib")

    # 4. Train Quantum VQC Model (Best Empirical Configuration: Exp B Feature-Selected VQC)
    print("\n--- Phase 7: Training Variational Quantum Classifier (VQC) ---")
    print(f"Applying Validated Improved Configuration (Exp B): 500 samples, 35% fraud weight, top 4 informative features.")
    
    X_train_q_sub, y_train_sub = get_quantum_training_subset(
        X_train_q, y_train, n_samples=500, fraud_weight=0.35, random_state=random_seed
    )

    start_q_time = time.time()
    quantum_model = QuantumFraudVQC(
        num_qubits=4,
        reps=1,
        entanglement="linear",
        optimizer_name="COBYLA",
        max_iter=cobyla_max_iter,
        random_seed=random_seed,
        shots=1024
    )
    quantum_model.fit(X_train_q_sub, y_train_sub)
    q_train_time = time.time() - start_q_time
    print(f"[Quantum VQC] Training completed in {q_train_time:.2f} seconds.")

    # 5. Evaluate Quantum Model on ALL 1,000 Unseen Test Samples
    print("\n--- Phase 8: Evaluating Quantum Model on ALL 1,000 Unseen Test Samples ---")
    q_pred = quantum_model.predict(X_test_q)
    q_prob = quantum_model.predict_proba(X_test_q)[:, 1]
    q_metrics = compute_model_metrics("Quantum VQC (Improved Exp B)", y_test, q_pred, q_prob)

    # Save quantum model artifacts
    quantum_model.save("models")

    # 6. Generate Comparison Table & Visualizations
    print("\n" + "=" * 70)
    print("                 MODEL PERFORMANCE EVALUATION")
    print("=" * 70)
    metrics_list = [rf_metrics, q_metrics]
    comp_df = generate_comparison_table(metrics_list, output_csv="results/metrics.csv")
    
    print("\nPerformance Comparison Table:")
    print(comp_df[["Metric", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].to_string(index=False))

    print("\nConfusion Matrix Breakdown:")
    print(f"Classical RF : TN={rf_metrics['TN']}, FP={rf_metrics['FP']}, FN={rf_metrics['FN']}, TP={rf_metrics['TP']}")
    print(f"Quantum VQC  : TN={q_metrics['TN']}, FP={q_metrics['FP']}, FN={q_metrics['FN']}, TP={q_metrics['TP']}")

    # Plot Confusion Matrices
    plot_side_by_side_confusion_matrices(
        rf_metrics["Confusion_Matrix"],
        q_metrics["Confusion_Matrix"],
        output_png="results/confusion_matrix.png"
    )

    # Plot ROC Curves
    plot_roc_curves(
        y_test,
        {
            "Random Forest (Classical)": rf_prob,
            "Quantum VQC (Best 4-Qubit)": q_prob
        },
        output_png="results/roc_curves.png"
    )

    print("\n[Pipeline] Training, evaluation, and artifact saving completed successfully!")
    return comp_df

if __name__ == "__main__":
    train_and_evaluate()
