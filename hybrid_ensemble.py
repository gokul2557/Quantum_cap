"""
hybrid_ensemble.py
------------------
Leakage-safe stacking procedure using 5-fold Out-Of-Fold (OOF) cross-validation predictions.
Tests whether quantum model predictions add predictive value to strong classical baselines
(Random Forest, XGBoost, Logistic Regression).
Evaluates strictly on the 1,000 unseen test samples.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

from preprocessing.preprocessing import load_and_clean_data, ALL_FEATURES

SEED = 42

def compute_metrics(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def main():
    print("=" * 80)
    print("   LEAKAGE-SAFE HYBRID ENSEMBLE INVESTIGATION (OUT-OF-FOLD STACKING)")
    print("=" * 80)

    # 1. Load Data
    df = load_and_clean_data("data/transactions.csv")
    df_X = df[ALL_FEATURES]
    df_y = df["fraud_label"]
    
    # 80/20 train/test split (hold out 1,000 test set completely)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, stratify=df_y, random_state=SEED
    )
    y_train = y_train.values
    y_test = y_test.values

    # Preprocess all 11 features for classical models
    scaler_full = StandardScaler()
    scaler_full.fit(X_train_df)
    X_train_full = scaler_full.transform(X_train_df)
    X_test_full = scaler_full.transform(X_test_df)

    # Preprocess top 4 features for Quantum Kernel model
    TOP4 = ["failed_attempts", "location_risk", "account_age_days", "merchant_risk"]
    scaler_q = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    scaler_q.fit(X_train_df[TOP4])
    X_train_q = scaler_q.transform(X_train_df[TOP4])
    X_test_q = scaler_q.transform(X_test_df[TOP4])

    # Precompute statevectors for the quantum fidelity kernel to allow fast 5-fold CV
    print("\nComputing Quantum Hilbert Space statevectors for training pool and test set...")
    zz_map = ZZFeatureMap(4, reps=1, entanglement="linear")
    t0 = time.time()
    train_sv = np.array([Statevector.from_instruction(zz_map.assign_parameters(x)).data for x in X_train_q])
    test_sv = np.array([Statevector.from_instruction(zz_map.assign_parameters(x)).data for x in X_test_q])
    print(f"Statevectors computed in {time.time() - t0:.2f} seconds.")

    # 2. Out-Of-Fold (OOF) Prediction Setup
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_rf = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_qsvm = np.zeros(len(y_train))

    test_preds_rf = np.zeros(len(y_test))
    test_preds_xgb = np.zeros(len(y_test))
    test_preds_qsvm = np.zeros(len(y_test))

    print(f"\nRunning {N_FOLDS}-Fold Leakage-Safe Out-of-Fold training...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
        print(f"  Training Fold {fold+1}/{N_FOLDS}...")
        # Fold partitions
        X_tr_f, y_tr_f = X_train_full[train_idx], y_train[train_idx]
        X_val_f, y_val_f = X_train_full[val_idx], y_train[val_idx]
        
        # 1. Classical Random Forest
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=SEED+fold, n_jobs=-1)
        rf_fold.fit(X_tr_f, y_tr_f)
        oof_rf[val_idx] = rf_fold.predict_proba(X_val_f)[:, 1]
        test_preds_rf += rf_fold.predict_proba(X_test_full)[:, 1] / N_FOLDS

        # 2. Classical XGBoost
        neg_pos = (y_tr_f == 0).sum() / (y_tr_f == 1).sum()
        xgb_fold = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=neg_pos, random_state=SEED+fold, eval_metric="logloss")
        xgb_fold.fit(X_tr_f, y_tr_f)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_val_f)[:, 1]
        test_preds_xgb += xgb_fold.predict_proba(X_test_full)[:, 1] / N_FOLDS

        # 3. Quantum Kernel SVM (on fold training)
        # Precomputed Gram matrix on fold
        sv_tr_f = train_sv[train_idx]
        sv_val_f = train_sv[val_idx]
        
        K_tr_fold = np.abs(sv_tr_f @ sv_tr_f.conj().T) ** 2
        K_val_fold = np.abs(sv_val_f @ sv_tr_f.conj().T) ** 2
        K_test_fold = np.abs(test_sv @ sv_tr_f.conj().T) ** 2

        qsvm_fold = SVC(kernel="precomputed", class_weight="balanced", probability=True, random_state=SEED+fold)
        qsvm_fold.fit(K_tr_fold, y_tr_f)
        oof_qsvm[val_idx] = qsvm_fold.predict_proba(K_val_fold)[:, 1]
        test_preds_qsvm += qsvm_fold.predict_proba(K_test_fold)[:, 1] / N_FOLDS

    print("\n--- OOF Individual Model PR-AUC & ROC-AUC Scores (Validation Ground Truth) ---")
    print(f"Classical RF  OOF PR-AUC: {average_precision_score(y_train, oof_rf):.4f} | ROC-AUC: {roc_auc_score(y_train, oof_rf):.4f}")
    print(f"Classical XGB OOF PR-AUC: {average_precision_score(y_train, oof_xgb):.4f} | ROC-AUC: {roc_auc_score(y_train, oof_xgb):.4f}")
    print(f"Quantum QSVM  OOF PR-AUC: {average_precision_score(y_train, oof_qsvm):.4f} | ROC-AUC: {roc_auc_score(y_train, oof_qsvm):.4f}")

    # 3. Stack Meta-Models: Classical-Only vs Hybrid (Classical + Quantum)
    # Meta features
    X_meta_classical_train = np.column_stack([oof_rf, oof_xgb])
    X_meta_classical_test = np.column_stack([test_preds_rf, test_preds_xgb])

    X_meta_hybrid_train = np.column_stack([oof_rf, oof_xgb, oof_qsvm])
    X_meta_hybrid_test = np.column_stack([test_preds_rf, test_preds_xgb, test_preds_qsvm])

    # Meta-Learner 1: Classical-Only Stacking
    meta_classical = LogisticRegression(class_weight="balanced", random_state=SEED)
    meta_classical.fit(X_meta_classical_train, y_train)
    val_prob_class = meta_classical.predict_proba(X_meta_classical_train)[:, 1]
    test_prob_class = meta_classical.predict_proba(X_meta_classical_test)[:, 1]
    test_pred_class = (test_prob_class >= 0.5).astype(int)

    # Meta-Learner 2: Hybrid (Classical + Quantum) Stacking
    meta_hybrid = LogisticRegression(class_weight="balanced", random_state=SEED)
    meta_hybrid.fit(X_meta_hybrid_train, y_train)
    val_prob_hyb = meta_hybrid.predict_proba(X_meta_hybrid_train)[:, 1]
    test_prob_hyb = meta_hybrid.predict_proba(X_meta_hybrid_test)[:, 1]
    test_pred_hyb = (test_prob_hyb >= 0.5).astype(int)

    # 4. Validation-Gate Rule Check
    val_pr_class = average_precision_score(y_train, val_prob_class)
    val_pr_hyb = average_precision_score(y_train, val_prob_hyb)
    
    val_roc_class = roc_auc_score(y_train, val_prob_class)
    val_roc_hyb = roc_auc_score(y_train, val_prob_hyb)

    print("\n" + "=" * 80)
    print("                      VALIDATION-GATE EVALUATION")
    print("=" * 80)
    print(f"Classical-Only Stack  Val PR-AUC: {val_pr_class:.4f} | Val ROC-AUC: {val_roc_class:.4f}")
    print(f"Hybrid (Q+Classical)  Val PR-AUC: {val_pr_hyb:.4f} | Val ROC-AUC: {val_roc_hyb:.4f}")
    print(f"Validation Gain Delta PR-AUC    : {val_pr_hyb - val_pr_class:+.4f}")
    print(f"Validation Gain Delta ROC-AUC   : {val_roc_hyb - val_roc_class:+.4f}")

    # Evaluate on Untouched 1,000-Sample Test Set
    m_class_test = compute_metrics("Classical Stack (RF + XGB)", y_test, test_pred_class, test_prob_class)
    m_hyb_test = compute_metrics("Hybrid Stack (Quantum + Classical)", y_test, test_pred_hyb, test_prob_hyb)

    ensemble_df = pd.DataFrame([m_class_test, m_hyb_test])
    ensemble_df.to_csv("results/experiments/ensemble_comparison_results.csv", index=False)

    print("\n" + "=" * 80)
    print("               FINAL HELD-OUT TEST EVALUATION (1,000 SAMPLES)")
    print("=" * 80)
    print(ensemble_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]].to_string(index=False))

    print("\nLearned Hybrid Meta-Weights in Logistic Stacking:")
    print(f"  Random Forest Weight : {meta_hybrid.coef_[0][0]:+.4f}")
    print(f"  XGBoost Weight       : {meta_hybrid.coef_[0][1]:+.4f}")
    print(f"  Quantum QSVM Weight  : {meta_hybrid.coef_[0][2]:+.4f}")

    if val_pr_hyb > val_pr_class and m_hyb_test["PR-AUC"] > m_class_test["PR-AUC"]:
        print("\n>> VERDICT: Quantum model adds validated predictive value to the ensemble.")
    else:
        print("\n>> VERDICT: Quantum model did NOT meaningfully improve over classical stacking.")
        print("   The classical ensemble already captures non-linear signals with high fidelity.")

if __name__ == "__main__":
    main()
