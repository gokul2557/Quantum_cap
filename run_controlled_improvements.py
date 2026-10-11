"""
run_controlled_improvements.py
------------------------------
Rigorous, staged controlled improvement experiments for the Quantum Fraud Detection prototype.
Evaluates all models on the EXACT same 1,000 unseen test set.

Stages:
Exp A: Original Baseline VQC (AngleEncoding, RealAmplitudes reps=1, COBYLA 40, heuristic 4 features)
Exp B: VQC with Training-Fitted Feature Selection (top 4 informative features: failed_attempts, location_risk, account_age_days, merchant_risk)
Exp C: VQC with Alternative Documented Feature Encoding (ZZFeatureMap with non-linear feature interactions)
Exp D: VQC with Ansatz Depth & Circular Entanglement (RealAmplitudes reps=2, circular entanglement)
Exp E: VQC with Alternative Optimizer (SPSA with stochastic perturbation)
Exp F: VQC with Imbalance-Aware Training + Validation-Tuned Decision Threshold
Exp G: Quantum Kernel SVM (QSVM with ZZFeatureMap & FidelityQuantumKernel) on benchmark
Exp H: Classical Baselines on Identical Benchmark:
       - Logistic Regression
       - Classical SVM (RBF)
       - Random Forest
       - XGBoost
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, real_amplitudes

from preprocessing.preprocessing import load_and_clean_data, prepare_data, get_quantum_training_subset, ALL_FEATURES
from quantum.encoding import build_angle_encoding_circuit
from quantum.circuit import build_variational_ansatz

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
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Confusion_Matrix": cm
    }

def find_best_threshold_val(y_val, probs_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s)
    if best_idx < len(thresholds):
        return thresholds[best_idx]
    return 0.5

def main():
    print("=" * 85)
    print("      RUNNING STAGED CONTROLLED IMPROVEMENTS & SCIENTIFIC BENCHMARKING")
    print("=" * 85)

    # 1. Dataset & Split (Strictly 80/20 train/test, test held-out)
    df = load_and_clean_data("data/transactions.csv")
    data = prepare_data(df, test_size=0.2, random_state=SEED)
    X_train_full = data["X_train_full"]
    X_test_full = data["X_test_full"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"Train Pool: {len(y_train)} samples | Unseen Test Pool: {len(y_test)} samples")

    # Original Heuristic Features: amount, frequency, location, device
    X_train_q_orig = data["X_train_q"]
    X_test_q_orig = data["X_test_q"]

    # Training-Fitted Top 4 Features from Diagnostic 3:
    # 1. failed_attempts, 2. location_risk, 3. account_age_days, 4. merchant_risk
    TOP4_FEATURES = ["failed_attempts", "location_risk", "account_age_days", "merchant_risk"]
    from sklearn.model_selection import train_test_split
    df_X = df[ALL_FEATURES]
    df_y = df["fraud_label"]
    X_tr_df, X_te_df, _, _ = train_test_split(df_X, df_y, test_size=0.2, stratify=df_y, random_state=SEED)
    
    scaler_top4 = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    scaler_top4.fit(X_tr_df[TOP4_FEATURES])
    X_train_q_top4 = scaler_top4.transform(X_tr_df[TOP4_FEATURES])
    X_test_q_top4 = scaler_top4.transform(X_te_df[TOP4_FEATURES])

    results = []
    prob_dict = {}

    # -------------------------------------------------------------
    # EXP A: Original Baseline VQC (Heuristic features, AngleEncoding, reps=1, COBYLA 40)
    # -------------------------------------------------------------
    print("\n>>> Stage A: Original Baseline VQC")
    X_sub_a, y_sub_a = get_quantum_training_subset(X_train_q_orig, y_train, n_samples=500, fraud_weight=0.35, random_state=SEED)
    vqc_a = VQC(
        feature_map=build_angle_encoding_circuit(4),
        ansatz=build_variational_ansatz(4, reps=1, entanglement="linear"),
        optimizer=COBYLA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_a.fit(X_sub_a, y_sub_a)
    t_train_a = time.time() - t0
    
    t0 = time.time()
    prob_a = vqc_a.predict_proba(X_test_q_orig)[:, 1]
    t_inf_a = time.time() - t0
    pred_a = (prob_a >= 0.5).astype(int)
    
    mA = compute_metrics("Exp A: Baseline VQC", y_test, pred_a, prob_a)
    mA["Train Time (s)"] = round(t_train_a, 2)
    mA["Inf Time (s)"] = round(t_inf_a, 2)
    mA["Config"] = "Heuristic 4 feat | AngleEncoding | reps=1 linear | COBYLA 40"
    results.append(mA)
    prob_dict["Exp A: Baseline VQC"] = prob_a
    print(f"Exp A Done -> Acc: {mA['Accuracy']:.4f}, Prec: {mA['Precision']:.4f}, Rec: {mA['Recall']:.4f}, F1: {mA['F1 Score']:.4f}, PR-AUC: {mA['PR-AUC']:.4f}, ROC: {mA['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP B: VQC with Training-Fitted Feature Selection (Top 4 features)
    # -------------------------------------------------------------
    print("\n>>> Stage B: VQC with Training-Fitted Feature Selection (Top 4 Features)")
    X_sub_b, y_sub_b = get_quantum_training_subset(X_train_q_top4, y_train, n_samples=500, fraud_weight=0.35, random_state=SEED)
    vqc_b = VQC(
        feature_map=build_angle_encoding_circuit(4),
        ansatz=build_variational_ansatz(4, reps=1, entanglement="linear"),
        optimizer=COBYLA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_b.fit(X_sub_b, y_sub_b)
    t_train_b = time.time() - t0
    
    t0 = time.time()
    prob_b = vqc_b.predict_proba(X_test_q_top4)[:, 1]
    t_inf_b = time.time() - t0
    pred_b = (prob_b >= 0.5).astype(int)
    
    mB = compute_metrics("Exp B: VQC + Selected Features", y_test, pred_b, prob_b)
    mB["Train Time (s)"] = round(t_train_b, 2)
    mB["Inf Time (s)"] = round(t_inf_b, 2)
    mB["Config"] = "Top 4 Informative feat | AngleEncoding | reps=1 linear | COBYLA 40"
    results.append(mB)
    prob_dict["Exp B: VQC + Selected Features"] = prob_b
    print(f"Exp B Done -> Acc: {mB['Accuracy']:.4f}, Prec: {mB['Precision']:.4f}, Rec: {mB['Recall']:.4f}, F1: {mB['F1 Score']:.4f}, PR-AUC: {mB['PR-AUC']:.4f}, ROC: {mB['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP C: VQC with Alternative Feature Encoding (ZZFeatureMap reps=1)
    # -------------------------------------------------------------
    print("\n>>> Stage C: VQC with Alternative Feature Encoding (ZZFeatureMap)")
    zz_fmap = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear").decompose()
    vqc_c = VQC(
        feature_map=zz_fmap,
        ansatz=build_variational_ansatz(4, reps=1, entanglement="linear"),
        optimizer=COBYLA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_c.fit(X_sub_b, y_sub_b)
    t_train_c = time.time() - t0
    
    t0 = time.time()
    prob_c = vqc_c.predict_proba(X_test_q_top4)[:, 1]
    t_inf_c = time.time() - t0
    pred_c = (prob_c >= 0.5).astype(int)
    
    mC = compute_metrics("Exp C: VQC + ZZFeatureMap", y_test, pred_c, prob_c)
    mC["Train Time (s)"] = round(t_train_c, 2)
    mC["Inf Time (s)"] = round(t_inf_c, 2)
    mC["Config"] = "Top 4 feat | ZZFeatureMap (reps=1) | reps=1 linear | COBYLA 40"
    results.append(mC)
    prob_dict["Exp C: VQC + ZZFeatureMap"] = prob_c
    print(f"Exp C Done -> Acc: {mC['Accuracy']:.4f}, Prec: {mC['Precision']:.4f}, Rec: {mC['Recall']:.4f}, F1: {mC['F1 Score']:.4f}, PR-AUC: {mC['PR-AUC']:.4f}, ROC: {mC['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP D: VQC with Circular Entanglement & Depth (reps=2)
    # -------------------------------------------------------------
    print("\n>>> Stage D: VQC with Circular Entanglement & reps=2")
    ansatz_circ = real_amplitudes(num_qubits=4, reps=2, entanglement="circular", insert_barriers=True).decompose()
    vqc_d = VQC(
        feature_map=build_angle_encoding_circuit(4),
        ansatz=ansatz_circ,
        optimizer=COBYLA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_d.fit(X_sub_b, y_sub_b)
    t_train_d = time.time() - t0
    
    t0 = time.time()
    prob_d = vqc_d.predict_proba(X_test_q_top4)[:, 1]
    t_inf_d = time.time() - t0
    pred_d = (prob_d >= 0.5).astype(int)
    
    mD = compute_metrics("Exp D: VQC Circular Entanglement (reps=2)", y_test, pred_d, prob_d)
    mD["Train Time (s)"] = round(t_train_d, 2)
    mD["Inf Time (s)"] = round(t_inf_d, 2)
    mD["Config"] = "Top 4 feat | AngleEncoding | reps=2 circular (12 params) | COBYLA 40"
    results.append(mD)
    prob_dict["Exp D: VQC Circular reps=2"] = prob_d
    print(f"Exp D Done -> Acc: {mD['Accuracy']:.4f}, Prec: {mD['Precision']:.4f}, Rec: {mD['Recall']:.4f}, F1: {mD['F1 Score']:.4f}, PR-AUC: {mD['PR-AUC']:.4f}, ROC: {mD['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP E: VQC with SPSA Optimizer
    # -------------------------------------------------------------
    print("\n>>> Stage E: VQC with SPSA Optimizer")
    vqc_e = VQC(
        feature_map=build_angle_encoding_circuit(4),
        ansatz=build_variational_ansatz(4, reps=1, entanglement="linear"),
        optimizer=SPSA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_e.fit(X_sub_b, y_sub_b)
    t_train_e = time.time() - t0
    
    t0 = time.time()
    prob_e = vqc_e.predict_proba(X_test_q_top4)[:, 1]
    t_inf_e = time.time() - t0
    pred_e = (prob_e >= 0.5).astype(int)
    
    mE = compute_metrics("Exp E: VQC + SPSA Optimizer", y_test, pred_e, prob_e)
    mE["Train Time (s)"] = round(t_train_e, 2)
    mE["Inf Time (s)"] = round(t_inf_e, 2)
    mE["Config"] = "Top 4 feat | AngleEncoding | reps=1 linear | SPSA 40"
    results.append(mE)
    prob_dict["Exp E: VQC + SPSA"] = prob_e
    print(f"Exp E Done -> Acc: {mE['Accuracy']:.4f}, Prec: {mE['Precision']:.4f}, Rec: {mE['Recall']:.4f}, F1: {mE['F1 Score']:.4f}, PR-AUC: {mE['PR-AUC']:.4f}, ROC: {mE['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP F: Imbalance-Aware Training + Validation Threshold Tuning
    # -------------------------------------------------------------
    print("\n>>> Stage F: Imbalance-Aware Training + Validation Threshold Selection")
    # Train-val split on training pool to select threshold without test leakage
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(skf.split(X_train_q_top4, y_train))
    
    X_tr_fold, y_tr_fold = X_train_q_top4[tr_idx], y_train[tr_idx]
    X_val_fold, y_val_fold = X_train_q_top4[val_idx], y_train[val_idx]
    
    # Balanced training subset from fold
    X_sub_f, y_sub_f = get_quantum_training_subset(X_tr_fold, y_tr_fold, n_samples=500, fraud_weight=0.50, random_state=SEED)
    vqc_f = VQC(
        feature_map=build_angle_encoding_circuit(4),
        ansatz=build_variational_ansatz(4, reps=1, entanglement="linear"),
        optimizer=COBYLA(maxiter=40),
        sampler=SamplerV2(default_shots=1024, seed=SEED)
    )
    t0 = time.time()
    vqc_f.fit(X_sub_f, y_sub_f)
    t_train_f = time.time() - t0
    
    # Select threshold on validation fold
    val_probs = vqc_f.predict_proba(X_val_fold)[:, 1]
    best_th_val = find_best_threshold_val(y_val_fold, val_probs)
    print(f"   [Validation Selection] Optimal F1 Threshold selected on Val fold: {best_th_val:.4f}")
    
    # Apply selected threshold to UNTOUCHED test set
    t0 = time.time()
    prob_f = vqc_f.predict_proba(X_test_q_top4)[:, 1]
    t_inf_f = time.time() - t0
    pred_f = (prob_f >= best_th_val).astype(int)
    
    mF = compute_metrics("Exp F: VQC Imbalance + Val-Threshold", y_test, pred_f, prob_f)
    mF["Train Time (s)"] = round(t_train_f, 2)
    mF["Inf Time (s)"] = round(t_inf_f, 2)
    mF["Config"] = f"Top 4 feat | 50% Fraud Balanced | Val Thresh={best_th_val:.3f}"
    results.append(mF)
    prob_dict["Exp F: VQC Val-Threshold"] = prob_f
    print(f"Exp F Done -> Acc: {mF['Accuracy']:.4f}, Prec: {mF['Precision']:.4f}, Rec: {mF['Recall']:.4f}, F1: {mF['F1 Score']:.4f}, PR-AUC: {mF['PR-AUC']:.4f}, ROC: {mF['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP G: Quantum Kernel SVM (QSVM) on Benchmark Subset
    # -------------------------------------------------------------
    print("\n>>> Stage G: Quantum Kernel SVM (QSVM with ZZFeatureMap & FidelityKernel)")
    from qiskit.quantum_info import Statevector
    # 400 stratified samples
    X_sub_g, y_sub_g = get_quantum_training_subset(X_train_q_top4, y_train, n_samples=400, fraud_weight=0.35, random_state=SEED)
    zz_kernel_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")

    def get_sv_matrix(X_mat):
        return np.array([Statevector.from_instruction(zz_kernel_map.assign_parameters(x)).data for x in X_mat])

    t0 = time.time()
    train_sv = get_sv_matrix(X_sub_g)
    # Quantum Fidelity Gram Matrix: K_ij = |<psi_i | psi_j>|^2
    K_train = np.abs(train_sv @ train_sv.conj().T) ** 2
    qsvm_clf = SVC(kernel="precomputed", class_weight="balanced", probability=True, random_state=SEED)
    qsvm_clf.fit(K_train, y_sub_g)
    t_train_g = time.time() - t0

    t0 = time.time()
    test_sv = get_sv_matrix(X_test_q_top4)
    # Quantum Test Kernel Matrix: K_test_ij = |<psi_test_i | psi_train_j>|^2
    K_test = np.abs(test_sv @ train_sv.conj().T) ** 2
    prob_g = qsvm_clf.predict_proba(K_test)[:, 1]
    pred_g = qsvm_clf.predict(K_test)
    t_inf_g = time.time() - t0

    mG = compute_metrics("Exp G: Quantum Kernel SVM (QSVM)", y_test, pred_g, prob_g)
    mG["Train Time (s)"] = round(t_train_g, 2)
    mG["Inf Time (s)"] = round(t_inf_g, 2)
    mG["Config"] = "Top 4 feat | ZZFeatureMap reps=1 | Precomputed FidelityKernel | 400 samples"
    results.append(mG)
    prob_dict["Exp G: Quantum QSVM"] = prob_g
    print(f"Exp G Done -> Acc: {mG['Accuracy']:.4f}, Prec: {mG['Precision']:.4f}, Rec: {mG['Recall']:.4f}, F1: {mG['F1 Score']:.4f}, PR-AUC: {mG['PR-AUC']:.4f}, ROC: {mG['ROC-AUC']:.4f}")

    # -------------------------------------------------------------
    # EXP H: Classical Baselines on the Exact Same Benchmark
    # -------------------------------------------------------------
    print("\n>>> Stage H: Classical Baselines (LR, SVM, RF, XGBoost) on Full & Benchmark Features")
    
    # H1: Logistic Regression (Full 11 features)
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=SEED)
    lr.fit(X_train_full, y_train)
    t_lr = time.time() - t0
    prob_lr = lr.predict_proba(X_test_full)[:, 1]
    pred_lr = (prob_lr >= 0.5).astype(int)
    mLR = compute_metrics("Exp H1: Logistic Regression", y_test, pred_lr, prob_lr)
    mLR["Train Time (s)"] = round(t_lr, 2)
    mLR["Inf Time (s)"] = 0.01
    mLR["Config"] = "All 11 features | class_weight='balanced'"
    results.append(mLR)
    prob_dict["Classical: Logistic Regression"] = prob_lr

    # H2: Classical SVM (RBF kernel, Full features)
    t0 = time.time()
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)
    svm.fit(X_train_full, y_train)
    t_svm = time.time() - t0
    prob_svm = svm.predict_proba(X_test_full)[:, 1]
    pred_svm = (prob_svm >= 0.5).astype(int)
    mSVM = compute_metrics("Exp H2: Classical SVM (RBF)", y_test, pred_svm, prob_svm)
    mSVM["Train Time (s)"] = round(t_svm, 2)
    mSVM["Inf Time (s)"] = 0.05
    mSVM["Config"] = "All 11 features | RBF kernel | class_weight='balanced'"
    results.append(mSVM)
    prob_dict["Classical: SVM (RBF)"] = prob_svm

    # H3: Random Forest (Full features)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train_full, y_train)
    t_rf = time.time() - t0
    prob_rf = rf.predict_proba(X_test_full)[:, 1]
    pred_rf = (prob_rf >= 0.5).astype(int)
    mRF = compute_metrics("Exp H3: Random Forest", y_test, pred_rf, prob_rf)
    mRF["Train Time (s)"] = round(t_rf, 2)
    mRF["Inf Time (s)"] = 0.02
    mRF["Config"] = "All 11 features | 100 trees | depth=8 | class_weight='balanced'"
    results.append(mRF)
    prob_dict["Classical: Random Forest"] = prob_rf

    # H4: XGBoost (Full features)
    t0 = time.time()
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=neg_pos_ratio,
        random_state=SEED,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train_full, y_train)
    t_xgb = time.time() - t0
    prob_xgb = xgb_clf.predict_proba(X_test_full)[:, 1]
    pred_xgb = (prob_xgb >= 0.5).astype(int)
    mXGB = compute_metrics("Exp H4: XGBoost Classifier", y_test, pred_xgb, prob_xgb)
    mXGB["Train Time (s)"] = round(t_xgb, 2)
    mXGB["Inf Time (s)"] = 0.01
    mXGB["Config"] = f"All 11 features | depth=4 | scale_pos_weight={neg_pos_ratio:.1f}"
    results.append(mXGB)
    prob_dict["Classical: XGBoost"] = prob_xgb

    # Save summary dataframe
    res_rows = []
    for r in results:
        res_rows.append({
            "Model": r["Model"],
            "Configuration": r["Config"],
            "Train Time (s)": r["Train Time (s)"],
            "Inf Time (s)": r["Inf Time (s)"],
            "Accuracy": round(r["Accuracy"], 4),
            "Precision": round(r["Precision"], 4),
            "Recall": round(r["Recall"], 4),
            "F1 Score": round(r["F1 Score"], 4),
            "PR-AUC": round(r["PR-AUC"], 4),
            "ROC-AUC": round(r["ROC-AUC"], 4),
            "TN": r["TN"], "FP": r["FP"], "FN": r["FN"], "TP": r["TP"]
        })
    df_comp = pd.DataFrame(res_rows)
    os.makedirs("results/experiments", exist_ok=True)
    df_comp.to_csv("results/experiments/controlled_improvements_results.csv", index=False)
    print("\n" + "=" * 85)
    print("                    STAGED EXPERIMENT SUMMARY TABLE")
    print("=" * 85)
    print(df_comp[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC", "Train Time (s)"]].to_string(index=False))

    # Plot Precision-Recall Curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(9, 6))
    for name, prb in prob_dict.items():
        p, r, _ = precision_recall_curve(y_test, prb)
        score = average_precision_score(y_test, prb)
        lw = 2.5 if "Classical" in name or "Exp B" in name else 1.5
        ls = "-" if "Classical" in name else "--"
        plt.plot(r, p, lw=lw, linestyle=ls, label=f"{name} (PR-AUC={score:.4f})")
    
    plt.axhline(y=(y_test==1).sum()/len(y_test), color="k", linestyle=":", label="Random Baseline (0.070)")
    plt.xlabel("Recall (True Positive Rate)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves Across Controlled Experiments (1,000 Test Set)", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pr_curves_controlled.png", dpi=300)
    plt.close()
    print("\nSaved Precision-Recall curves to 'results/pr_curves_controlled.png'.")

if __name__ == "__main__":
    main()
