# ⚛️ Quantum AI-Based Real-Time Fraud Detection System

**Hybrid Quantum-Classical Financial Transaction Monitoring | Academic Prototype**

---

## 📌 Problem Statement

> **“Design a Quantum AI-based Fraud Detection System to identify suspicious financial transactions in real time.”**

Financial institutions process millions of transactions per second. Fraud detection is notoriously difficult due to extreme class imbalance (~5–10% fraud) and non-linear interactions across high-dimensional features. This academic prototype demonstrates how a **Variational Quantum Classifier (VQC)** running on a quantum simulator can be integrated with classical feature pipelines to perform real-time financial transaction scoring and classification.

---

## 🎯 Objective

1. Implement a genuine **Quantum Machine Learning (QML)** model using **Qiskit** and **Qiskit Machine Learning** (no faked or mocked quantum processing).
2. Encode classical financial indicators into quantum states via **Angle Feature Encoding**.
3. Correlate features across qubits using a parameterized **CNOT entangling circuit**.
4. Benchmark the quantum model honestly against a classical **Random Forest** baseline.
5. Provide an interactive **Streamlit dashboard** for live real-time demonstrations, manual transaction evaluation, and transaction history logging.

---

## 🏗️ System Architecture

```text
                 FINANCIAL TRANSACTION
                          │
                          ▼
                ┌─────────────────────┐
                │  Transaction Input  │ (Manual Form / Simulation)
                └─────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Data Preprocessing  │ (Handling nulls, duplicates)
                │ MinMax / Std Scaling│ (Fitted strictly on train data)
                └─────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Feature Selection  │ (4 Key Quantum Features)
                └─────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Quantum Feature     │ Angle Encoding:
                │ Encoding            │ Ry(θ0), Ry(θ1), Ry(θ2), Ry(θ3)
                └─────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Quantum ML Model   │ Parameterized Ansatz + CNOT
                │  VQC (Qiskit ML)    │ StatevectorSampler (Aer)
                └─────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Fraud Risk Score   │ (0.0% to 100.0%)
                └─────────────────────┘
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
          LEGITIMATE           SUSPICIOUS
         (Risk < 30%)         (Risk ≥ 30%)
                 │                 │
                 ▼                 ▼
         APPROVE TXN          FLAG / ALERT
```

---

## 💻 Technology Stack

- **Language:** Python 3.13 / 3.11+
- **Quantum Computing:**
  - `qiskit` (2.4.2)
  - `qiskit-aer` (0.17.2)
  - `qiskit-machine-learning` (0.9.1)
- **Classical Machine Learning:** `scikit-learn` (1.9.0)
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Web Interface:** `streamlit` (1.65.0)
- **Persistence:** `joblib`, `dill`

---

## 📊 Dataset: Synthetic Financial Transactions

- **Title:** *Synthetic Financial Transaction Dataset for Academic Prototype*
- **Total Transactions:** 5,000
- **Class Imbalance:** 93.0% Legitimate (`fraud_label=0`), 7.0% Fraudulent (`fraud_label=1`).
- **Features (12 Total):**
  1. `transaction_amount`: ₹50 to ₹150,000 (log-normal distribution)
  2. `transaction_frequency`: 1 to 30 transactions/day
  3. `location_risk`: 0=Low, 1=Medium, 2=High
  4. `device_changed`: 0=No, 1=Yes
  5. `previous_fraud`: 0=No, 1=Yes
  6. `distance_from_usual_location`: 0.1 to 2,500 km
  7. `transaction_hour`: 0 to 23 (diurnal pattern)
  8. `merchant_risk`: 0=Low, 1=Medium, 2=High
  9. `failed_attempts`: 0 to 4 prior PIN/OTP failures
  10. `international_transaction`: 0=No, 1=Yes
  11. `account_age_days`: 10 to 3,650 days
  12. `fraud_label`: 0=Legitimate, 1=Fraudulent

*Notice: Transactions contain realistic stochastic noise and feature correlations, ensuring machine learning is genuinely required.*

---

## 🔬 Quantum Architecture Details

### 1. Selected Quantum Features (4 Qubits)
Current NISQ simulators on consumer laptops are optimal with compact qubit registers. We select 4 primary risk drivers:
- $q_0$: `transaction_amount`
- $q_1$: `transaction_frequency`
- $q_2$: `location_risk`
- $q_3$: `device_changed`

### 2. Angle Encoding
Continuous features are scaled to $[0, \pi]$ using `MinMaxScaler` fitted solely on training data. Each feature controls the rotation angle of an $R_y$ gate:
$$\vert \psi(x) \rangle = \bigotimes_{i=0}^3 R_y(x_i) \vert 0 \rangle$$

### 3. Entangling Variational Circuit
- **Ansatz:** Hardware-efficient alternating $R_y(w_j)$ rotations and linear CNOT entanglement:
  $$q_0 \to q_1 \to q_2 \to q_3$$
- **Parameters:** 8 variational weights.
- **Optimizer:** Classical `COBYLA` optimizer.
- **Backend:** `StatevectorSampler` local quantum simulator.

---

## 📈 Measured Model Performance (Fair 1,000-Sample Test Set Evaluation)

Both models were evaluated on the **exact same 1,000 unseen test transactions** (930 Legitimate, 70 Fraudulent).

| Metric | Classical Baseline (Random Forest) | Original Baseline VQC (Heuristic 4 feat) | Improved Quantum VQC (Exp B: Feature-Selected) |
|---|:---:|:---:|:---:|
| **Train Samples** | 4,000 (Full) | 500 | 500 (Top 4 Informative) |
| **Accuracy** | 91.40% | 90.60% | **91.70%** |
| **Precision** | 43.65% | 14.71% | **42.53%** |
| **Recall (Fraud Catch Rate)** | **78.57%** | 7.14% | **52.86%** |
| **F1 Score** | **0.5612** | 0.0962 | **0.4713 (+390%)** |
| **ROC-AUC** | **0.9498** | 0.5152 | **0.8212 (+59%)** |
| **PR-AUC** | **0.6819** | 0.1000 | **0.4803 (+380%)** |
| **Training Time** | 1.82s | 90.37s | 122.47s |

### Confusion Matrix Breakdown (Exact Same 1,000 Unseen Test Transactions)
- **Random Forest:** True Negatives (TN): **859** | False Positives (FP): **71** | False Negatives (FN): **15** | True Positives (TP): **55**
- **Original Baseline VQC:** True Negatives (TN): **901** | False Positives (FP): **29** | False Negatives (FN): **65** | True Positives (TP): **5**
- **Improved Quantum VQC (Exp B):** True Negatives (TN): **880** | False Positives (FP): **50** | False Negatives (FN): **33** | True Positives (TP): **37**

### 🔬 Controlled Diagnostic & Scientific Improvement Findings
A comprehensive diagnostic and staged improvement study was performed:
1. **Diagnosis (Feature Starvation):** The original heuristic 4 features captured only 36.97% of dataset predictive signal. Replacing them with the training-fitted top 4 features (`failed_attempts`, `location_risk`, `account_age_days`, `merchant_risk`) triggered a massive performance leap: **PR-AUC jumped from 0.1000 to 0.4803 (+380%)** and **F1 jumped from 0.0962 to 0.4713 (+390%)**.
2. **Threshold Tuning on Validation Data (Exp F):** Selecting decision thresholds on validation cross-validation folds elevated fraud recall to **70.00%** while maintaining 88.60% accuracy on untouched test data.
3. **Leakage-Safe Hybrid Stacking:** Out-of-fold cross-validation stacking revealed that while a Quantum Kernel SVM adds non-zero weight (+0.372) to meta-learning, classical gradient boosting (XGBoost) and Random Forest already capture tabular interactions thoroughly, producing identical test PR-AUC (0.7498).
4. **Honest Assessment (No Quantum Advantage):** Classical Random Forest and XGBoost achieve higher PR-AUC (0.6819–0.7607) and train in 1–2 seconds. However, the improved 4-qubit VQC demonstrates that with proper feature selection, QML achieves genuine, non-trivial fraud ranking (**ROC-AUC 0.8212, PR-AUC 0.4803**) on simulated quantum hardware.

---

## 🚀 Installation & Running

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset and Train Models
```bash
python train_models.py
```
*(Preprocesses data, trains Random Forest & Quantum VQC, and saves evaluation reports).*

### Step 3: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎭 Live Demonstration Scenarios

### Demo 1 — Normal / Legitimate Transaction
- **Amount:** ₹1,200
- **Frequency:** 2/day
- **Location Risk:** Low (0)
- **Device Changed:** No (0)
- **Previous Fraud:** No (0)
- **Distance:** 3 km
- **Expected Result:** **Low Risk / Legitimate** (~30–35% score), Action: `APPROVE TRANSACTION`.

### Demo 2 — Suspicious / High-Risk Transaction
- **Amount:** ₹75,000
- **Frequency:** 16/day
- **Location Risk:** High (2)
- **Device Changed:** Yes (1)
- **Previous Fraud:** Yes (1)
- **Distance:** 850 km
- **Transaction Hour:** 2:00 AM
- **Expected Result:** **Suspicious / High Risk** (> 69–99% score), Action: `FLAG TRANSACTION FOR VERIFICATION`.

---

## 🎤 2–3 Minute Live Presentation Script

> **"Respected professors and evaluators,**
>
> Today we present a prototype for **Quantum AI-Based Real-Time Financial Fraud Detection**.
>
> **1. The Challenge:**  
> Fraud detection requires analyzing complex, interdependent transaction patterns in milliseconds, where fraudulent events represent less than 10% of total volume.
>
> **2. Our Architecture:**  
> We built a hybrid classical-quantum framework. Classical preprocessing handles scaling and non-leakage train/test splits. We then perform **Angle Feature Encoding** to map normalized financial features—such as transaction amount, velocity, location risk, and device changes—into quantum state rotations on a 4-qubit register.
>
> **3. Quantum AI Component:**  
> Using **Qiskit and Qiskit Machine Learning**, our system executes a **Variational Quantum Classifier (VQC)** with linear CNOT entanglement and parameterized $R_y$ rotations on a local quantum simulator.
>
> **4. Live Demonstration:**  
> When a new transaction arrives—either entered manually or generated via our real-time simulator—it is evaluated through our quantum circuit. The system computes a calibrated **Fraud Risk Score**, classifies the transaction as Legitimate or Suspicious, recommends an immediate action, and provides transparent Transaction Risk Indicators.
>
> **5. Honest Findings:**  
> We benchmarked the VQC against a classical Random Forest. While classical ensembles are currently superior on tabular data, this project proves the end-to-end viability of QML for transaction monitoring as quantum hardware advances toward fault tolerance."

---

## ⚠️ Limitations & Future Work

### Prototype Limitations
1. Executes on a classical statevector simulator (not physical cryogenic quantum hardware).
2. Restricted to 4 qubits to allow interactive latency on normal laptops.
3. Evaluates 4 quantum features compared to 11 classical features.
4. Uses synthetic academic data rather than confidential banking feeds.

### Future Work
- Integration with IBM Quantum cloud hardware (QPUs).
- Higher-dimensional amplitude encoding or quantum kernels (QSVM).
- Kafka streaming integration for high-throughput banking channels.
