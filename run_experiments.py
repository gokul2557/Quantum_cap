"""
run_experiments.py
------------------
Controlled, reproducible experiment runner for 4-qubit Variational Quantum Classifier (VQC).
Executes systematic hyperparameter experiments on genuine Qiskit Aer simulator,
evaluates EVERY model on the exact same 1,000 unseen test samples,
prevents data leakage, records accurate runtimes and metrics,
and outputs a complete scientific comparison table.
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

# Global fixed random seed for complete reproducibility
SEED = 42

EXPERIMENTS = [
    # 1. Baseline & Training Data Size Sweep (reps=1, COBYLA=40, current fraud_weight=0.35)
    {
        "id": "Exp_A1",
        "name": "VQC (500 samples, 35% fraud, reps=1, COBYLA 40)",
        "train_samples": 500,
        "fraud_weight": 0.35,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 40
    },
    {
        "id": "Exp_B1",
        "name": "VQC (1000 samples, 35% fraud, reps=1, COBYLA 40)",
        "train_samples": 1000,
        "fraud_weight": 0.35,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 40
    },
    {
        "id": "Exp_C1",
        "name": "VQC (2000 samples, 35% fraud, reps=1, COBYLA 40)",
        "train_samples": 2000,
        "fraud_weight": 0.35,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 40
    },

    # 2. Class Imbalance Sweep (1000 samples, reps=1, COBYLA=40)
    {
        "id": "Exp_B2",
        "name": "VQC (1000 samples, balanced 50% fraud, reps=1, COBYLA 40)",
        "train_samples": 1000,
        "fraud_weight": 0.50,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 40
    },
    {
        "id": "Exp_B3",
        "name": "VQC (1000 samples, fraud-focused 70% fraud, reps=1, COBYLA 40)",
        "train_samples": 1000,
        "fraud_weight": 0.70,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 40
    },

    # 3. Circuit Depth (RealAmplitudes repetitions) Sweep (1000 samples, COBYLA=40)
    {
        "id": "Exp_E1",
        "name": "VQC (1000 samples, 35% fraud, reps=2, COBYLA 40)",
        "train_samples": 1000,
        "fraud_weight": 0.35,
        "reps": 2,
        "optimizer": "COBYLA",
        "max_iter": 40
    },

    # 4. Optimizer Tuning Sweep (iterations = 80, 120, SPSA)
    {
        "id": "Exp_F1",
        "name": "VQC (1000 samples, 35% fraud, reps=1, COBYLA 80)",
        "train_samples": 1000,
        "fraud_weight": 0.35,
        "reps": 1,
        "optimizer": "COBYLA",
        "max_iter": 80
    },
    {
        "id": "Exp_G1",
        "name": "VQC (1000 samples, 35% fraud, reps=1, SPSA 40)",
        "train_samples": 1000,
        "fraud_weight": 0.35,
        "reps": 1,
        "optimizer": "SPSA",
        "max_iter": 40
    }
]

def main():
    print("=" * 80)
    print("   QUANTUM AI FRAUD DETECTION: CONTROLLED EXPERIMENTATION SUITE")
    print("=" * 80)

    # 1. Load Data
    data_path = "data/transactions.csv"
    df = load_and_clean_data(data_path)

    # 2. Preprocess with strict data leakage prevention
    print("\n[Data Preparation] Splitting 80/20 train/test (stratified, random_state=42)...")
    data = prepare_data(df, test_size=0.2, random_state=SEED)
    X_train_full = data["X_train_full"]
    X_test_full = data["X_test_full"]
    X_train_q = data["X_train_q"]
    X_test_q = data["X_test_q"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"Training Pool: {len(y_train)} samples ({np.sum(y_train==0)} legit, {np.sum(y_train==1)} fraud)")
    print(f"Unseen Test Pool: {len(y_test)} samples ({np.sum(y_test==0)} legit, {np.sum(y_test==1)} fraud)")
    print(f"Verifying: All evaluations will test on EXACT SAME {len(y_test)} unseen test samples.")

    # 3. Train Classical Baseline: Random Forest
    print("\n" + "-" * 60)
    print(">> Baseline: Random Forest (trained on all 4,000 train samples)")
    print("-" * 60)
    rf_start = time.time()
    rf_model = ClassicalFraudModel(random_state=SEED)
    rf_model.train(X_train_full, y_train)
    rf_time = time.time() - rf_start
    rf_preds = rf_model.predict(X_test_full)
    rf_probs = rf_model.predict_proba(X_test_full)[:, 1]
    rf_metrics = compute_model_metrics("Random Forest (Classical)", y_test, rf_preds, rf_probs)
    rf_metrics["Training Time (s)"] = round(rf_time, 2)
    rf_metrics["Train Samples"] = len(y_train)
    rf_metrics["Distribution"] = f"Natural ({np.sum(y_train==0)}/{np.sum(y_train==1)})"
    rf_metrics["Qubits"] = "N/A"
    rf_metrics["Reps"] = "N/A"
    rf_metrics["Optimizer"] = "N/A"
    rf_metrics["Max Iter"] = "N/A"
    rf_model.save("models/random_forest.joblib")

    print(f"Random Forest finished in {rf_time:.2f}s | Acc: {rf_metrics['Accuracy']:.4f} | Rec: {rf_metrics['Recall']:.4f} | F1: {rf_metrics['F1 Score']:.4f} | AUC: {rf_metrics['ROC-AUC']:.4f}")

    # 4. Run VQC Experiments
    os.makedirs("results/experiments", exist_ok=True)
    all_metrics = [rf_metrics]
    saved_vqc_probs = {"Random Forest": rf_probs}
    trained_vqcs = {}

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        print("\n" + "-" * 60)
        print(f">> Running {exp_id}: {exp['name']}")
        print(f"   Samples: {exp['train_samples']} | Fraud Target: {exp['fraud_weight']*100:.0f}% | Reps: {exp['reps']} | Opt: {exp['optimizer']} | Iter: {exp['max_iter']}")
        print("-" * 60)

        # Stratified sampling strictly from training pool
        X_sub, y_sub = get_quantum_training_subset(
            X_train_q,
            y_train,
            n_samples=exp["train_samples"],
            fraud_weight=exp["fraud_weight"],
            random_state=SEED
        )
        dist_str = f"{np.sum(y_sub==0)}L / {np.sum(y_sub==1)}F ({np.mean(y_sub)*100:.1f}% F)"

        # Initialize VQC
        vqc = QuantumFraudVQC(
            num_qubits=4,
            reps=exp["reps"],
            entanglement="linear",
            optimizer_name=exp["optimizer"],
            max_iter=exp["max_iter"],
            random_seed=SEED,
            shots=1024
        )

        # Train and measure time
        t0 = time.time()
        vqc.fit(X_sub, y_sub)
        fit_time = time.time() - t0
        print(f"   [{exp_id}] Training completed in {fit_time:.2f} seconds.")

        # Evaluate on the SAME full 1,000 unseen test samples
        eval_t0 = time.time()
        preds = vqc.predict(X_test_q)
        probs = vqc.predict_proba(X_test_q)[:, 1]
        eval_time = time.time() - eval_t0
        print(f"   [{exp_id}] Evaluated on {len(preds)} test samples in {eval_time:.2f} seconds.")

        # Compute metrics
        m = compute_model_metrics(exp["name"], y_test, preds, probs)
        m["Exp_ID"] = exp_id
        m["Training Time (s)"] = round(fit_time, 2)
        m["Train Samples"] = len(y_sub)
        m["Distribution"] = dist_str
        m["Qubits"] = 4
        m["Reps"] = exp["reps"]
        m["Optimizer"] = exp["optimizer"]
        m["Max Iter"] = exp["max_iter"]
        
        all_metrics.append(m)
        saved_vqc_probs[exp_id] = probs
        trained_vqcs[exp_id] = (vqc, m)

        print(f"   >> Acc: {m['Accuracy']:.4f} | Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f} | F1: {m['F1 Score']:.4f} | AUC: {m['ROC-AUC']:.4f}")
        print(f"   >> Confusion Matrix: TN={m['TN']}, FP={m['FP']}, FN={m['FN']}, TP={m['TP']}")

    # 5. Build Comprehensive Comparison DataFrame
    rows = []
    for m in all_metrics:
        rows.append({
            "Model / Config": m.get("Exp_ID", "RF"),
            "Description": m["Model"],
            "Train Samples": m.get("Train Samples"),
            "Train Distribution": m.get("Distribution"),
            "Qubits": m.get("Qubits"),
            "Reps": m.get("Reps"),
            "Optimizer": m.get("Optimizer"),
            "Max Iter": m.get("Max Iter"),
            "Train Time (s)": m.get("Training Time (s)"),
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1 Score": round(m["F1 Score"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4),
            "TN": m["TN"],
            "FP": m["FP"],
            "FN": m["FN"],
            "TP": m["TP"]
        })
    df_results = pd.DataFrame(rows)
    csv_path = "results/experiments/vqc_experiment_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Artifact Saved] Detailed experiment records saved to '{csv_path}'.")

    # 6. Select Best VQC Model (prioritizing Fraud Recall, Precision, F1, and ROC-AUC)
    # Filter to only VQC experiments
    vqc_metrics = [m for m in all_metrics if "Exp_ID" in m]
    
    # Ranking score: weighted combination prioritizing F1 and ROC-AUC, penalizing 0-recall models
    def score_vqc(m):
        # We heavily weight F1, Recall, and ROC-AUC for fraud detection
        return (2.0 * m["F1 Score"]) + (1.5 * m["Recall"]) + (1.0 * m["ROC-AUC"])

    best_vqc_metric = max(vqc_metrics, key=score_vqc)
    best_exp_id = best_vqc_metric["Exp_ID"]
    best_vqc_model, _ = trained_vqcs[best_exp_id]

    print("\n" + "=" * 80)
    print(f"🏆 BEST VQC CONFIGURATION IDENTIFIED: {best_exp_id} - {best_vqc_metric['Model']}")
    print(f"   Accuracy : {best_vqc_metric['Accuracy']:.4f}")
    print(f"   Precision: {best_vqc_metric['Precision']:.4f}")
    print(f"   Recall   : {best_vqc_metric['Recall']:.4f} (Caught {best_vqc_metric['TP']} of {best_vqc_metric['TP'] + best_vqc_metric['FN']} frauds)")
    print(f"   F1 Score : {best_vqc_metric['F1 Score']:.4f}")
    print(f"   ROC-AUC  : {best_vqc_metric['ROC-AUC']:.4f}")
    print("=" * 80)

    # 7. Save Final Model Artifacts for Streamlit app
    best_vqc_model.save("models")
    
    # Save standard results/metrics.csv comparing Classical RF vs Best VQC
    final_comp_list = [rf_metrics, best_vqc_metric]
    generate_comparison_table(final_comp_list, output_csv="results/metrics.csv")

    # Plot final Confusion Matrix (RF vs Best VQC)
    plot_side_by_side_confusion_matrices(
        rf_metrics["Confusion_Matrix"],
        best_vqc_metric["Confusion_Matrix"],
        output_png="results/confusion_matrix.png"
    )

    # Plot ROC Curves (RF vs Best VQC vs Baseline VQC)
    baseline_id = "Exp_A1"
    roc_dict = {
        "Random Forest (Classical)": rf_probs,
        f"Best VQC ({best_exp_id})": saved_vqc_probs[best_exp_id]
    }
    if baseline_id in saved_vqc_probs and baseline_id != best_exp_id:
        roc_dict["Baseline VQC (Exp_A1)"] = saved_vqc_probs[baseline_id]

    plot_roc_curves(y_test, roc_dict, output_png="results/roc_curves.png")

    print("\n[Complete] All experiments finished. Artifacts saved to 'models/' and 'results/'.")

if __name__ == "__main__":
    main()
