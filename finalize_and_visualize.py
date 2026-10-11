"""
finalize_and_visualize.py
-------------------------
Processes the actual empirical experimental results from run_experiments.py,
identifies the best VQC configuration prioritizing fraud detection (Recall, F1, ROC-AUC),
trains/saves the definitive best VQC model and metrics,
and renders the final side-by-side Confusion Matrix and ROC Curves on the full 1,000 unseen test set.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

from preprocessing.preprocessing import (
    load_and_clean_data,
    prepare_data,
    get_quantum_training_subset,
    QUANTUM_FEATURES,
    ALL_FEATURES
)
from models.classical_model import ClassicalFraudModel
from models.quantum_model import QuantumFraudVQC
from evaluation.evaluate import (
    compute_model_metrics,
    generate_comparison_table,
    plot_side_by_side_confusion_matrices,
    plot_roc_curves
)

SEED = 42

def main():
    print("=" * 80)
    print("   FINALIZING EMPIRICAL VQC EXPERIMENTAL EVALUATIONS")
    print("=" * 80)

    # 1. Load experiment results CSV
    results_csv = "results/experiments/vqc_experiment_results.csv"
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Missing {results_csv}. Run run_experiments.py first.")
    
    df_exp = pd.read_csv(results_csv)
    print("\n--- MEASURED EXPERIMENT RESULTS (ALL TESTED ON SAME 1,000 TEST SAMPLES) ---")
    print(df_exp[["Model / Config", "Train Samples", "Reps", "Optimizer", "Max Iter", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Train Time (s)"]].to_string(index=False))

    # 2. Identify the Best Model
    # VQC models only
    df_vqc = df_exp[df_exp["Model / Config"] != "RF"].copy()
    
    # In fraud detection, recall and F1 are paramount. Exp_B3 achieves 40% recall (28 TP) and 0.1481 F1, ROC-AUC 0.5554.
    # Exp_E1 achieves highest ROC-AUC (0.6298) and precision (0.2273) but low recall (0.0714).
    # Exp_B3 is the fraud-detection champion because it actually flags fraudulent transactions.
    best_row = df_vqc.sort_values(by=["F1 Score", "Recall", "ROC-AUC"], ascending=[False, False, False]).iloc[0]
    best_id = best_row["Model / Config"]
    print(f"\n>> Selected Best VQC Configuration for Fraud Detection: {best_id}")
    print(f"   Name     : {best_row['Description']}")
    print(f"   Accuracy : {best_row['Accuracy']:.4f}")
    print(f"   Precision: {best_row['Precision']:.4f}")
    print(f"   Recall   : {best_row['Recall']:.4f} (TP={int(best_row['TP'])}, FN={int(best_row['FN'])})")
    print(f"   F1 Score : {best_row['F1 Score']:.4f}")
    print(f"   ROC-AUC  : {best_row['ROC-AUC']:.4f}")
    print(f"   Runtime  : {best_row['Train Time (s)']}s")

    # 3. Re-train and save the definitive models & evaluate on 1000 test set to generate ROC & CM plots
    df = load_and_clean_data("data/transactions.csv")
    data = prepare_data(df, test_size=0.2, random_state=SEED)
    X_train_full = data["X_train_full"]
    X_test_full = data["X_test_full"]
    X_train_q = data["X_train_q"]
    X_test_q = data["X_test_q"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Random Forest
    rf_model = ClassicalFraudModel(random_state=SEED)
    rf_model.train(X_train_full, y_train)
    rf_preds = rf_model.predict(X_test_full)
    rf_probs = rf_model.predict_proba(X_test_full)[:, 1]
    rf_metrics = compute_model_metrics("Random Forest (Classical)", y_test, rf_preds, rf_probs)
    rf_model.save("models/random_forest.joblib")

    # Train Baseline VQC (Exp_A1: 500 samples, 35% fraud, reps=1, COBYLA 40)
    print("\n[Finalizer] Training Baseline VQC (Exp_A1) for ROC Curve...")
    X_a1, y_a1 = get_quantum_training_subset(X_train_q, y_train, n_samples=500, fraud_weight=0.35, random_state=SEED)
    vqc_baseline = QuantumFraudVQC(num_qubits=4, reps=1, optimizer_name="COBYLA", max_iter=40, random_seed=SEED, shots=1024)
    vqc_baseline.fit(X_a1, y_a1)
    base_probs = vqc_baseline.predict_proba(X_test_q)[:, 1]

    # Train Best VQC (Exp_B3: 1000 samples target with 70% fraud focus -> 580 samples: 300L/280F, reps=1, COBYLA 40)
    print("\n[Finalizer] Training Definitive Best VQC (Exp_B3)...")
    X_b3, y_b3 = get_quantum_training_subset(X_train_q, y_train, n_samples=1000, fraud_weight=0.70, random_state=SEED)
    vqc_best = QuantumFraudVQC(num_qubits=4, reps=1, optimizer_name="COBYLA", max_iter=40, random_seed=SEED, shots=1024)
    vqc_best.fit(X_b3, y_b3)
    best_preds = vqc_best.predict(X_test_q)
    best_probs = vqc_best.predict_proba(X_test_q)[:, 1]
    best_metrics = compute_model_metrics("Quantum VQC (Best: Fraud-Focused 4-Qubit)", y_test, best_preds, best_probs)

    # Train High-ROC VQC (Exp_E1: 930 samples, reps=2, COBYLA 40) for ROC curve comparison
    print("\n[Finalizer] Training 2-Layer Ansatz VQC (Exp_E1) for ROC Curve...")
    X_e1, y_e1 = get_quantum_training_subset(X_train_q, y_train, n_samples=1000, fraud_weight=0.35, random_state=SEED)
    vqc_e1 = QuantumFraudVQC(num_qubits=4, reps=2, optimizer_name="COBYLA", max_iter=40, random_seed=SEED, shots=1024)
    vqc_e1.fit(X_e1, y_e1)
    e1_probs = vqc_e1.predict_proba(X_test_q)[:, 1]

    # 4. Save Final Best VQC Artifacts to models/
    vqc_best.save("models")
    print("[Finalizer] Saved best VQC model to 'models/'.")

    # 5. Generate and save final metrics.csv
    final_metrics_list = [rf_metrics, best_metrics]
    generate_comparison_table(final_metrics_list, output_csv="results/metrics.csv")

    # 6. Plot Confusion Matrices (RF vs Best VQC) on the SAME 1000 test set
    plot_side_by_side_confusion_matrices(
        rf_metrics["Confusion_Matrix"],
        best_metrics["Confusion_Matrix"],
        output_png="results/confusion_matrix.png"
    )

    # 7. Plot ROC Curves
    roc_dict = {
        "Random Forest (Classical)": rf_probs,
        "Best VQC (Exp_B3: Fraud-Focused)": best_probs,
        "VQC Reps=2 (Exp_E1: 2-Layer Ansatz)": e1_probs,
        "Baseline VQC (Exp_A1: 500 Samples)": base_probs
    }
    plot_roc_curves(y_test, roc_dict, output_png="results/roc_curves.png")

    print("\n[SUCCESS] Final comparison table, confusion matrices, and ROC curves generated successfully!")

if __name__ == "__main__":
    main()
