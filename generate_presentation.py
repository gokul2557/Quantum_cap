"""
generate_presentation.py
------------------------
Generates an updated, fully-aligned PowerPoint presentation (.pptx)
matching the student's project theme, QSVM algorithm, ZZ Feature Map,
and filled-in performance evaluation metrics.
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_pptx: str = "Quantum_AI_Fraud_Detection_Presentation.pptx"):
    prs = pptx.Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme Colors
    BG_DARK = RGBColor(26, 11, 46)       # Deep Purple / Indigo
    CARD_BG = RGBColor(45, 27, 78)       # Lighter Purple Card
    ACCENT_CYAN = RGBColor(0, 229, 255)  # Cyan Accent
    ACCENT_PURPLE = RGBColor(168, 85, 247) # Bright Purple
    TEXT_WHITE = RGBColor(255, 255, 255)
    TEXT_MUTED = RGBColor(209, 213, 219)
    GREEN = RGBColor(52, 211, 153)
    RED = RGBColor(248, 113, 113)

    blank_layout = prs.slide_layouts[6]

    def add_bg(slide):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = BG_DARK
        shape.line.fill.background()
        return shape

    def add_card(slide, left, top, width, height, title=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = RGBColor(79, 46, 128)
        card.line.width = Pt(1.5)
        
        if title:
            txBox = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), width - Inches(0.6), Inches(0.6))
            tf = txBox.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.bold = True
            p.font.size = Pt(20)
            p.font.color.rgb = ACCENT_CYAN
        return card

    # =========================================================
    # SLIDE 1: Title Slide
    # =========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    add_bg(slide1)
    
    tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Quantum AI-Based Fraud\nDetection System"
    p.font.bold = True
    p.font.size = Pt(44)
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(24)
    
    p2 = tf1.add_paragraph()
    p2.text = "Identifying suspicious financial transactions in real time"
    p2.font.size = Pt(24)
    p2.font.color.rgb = ACCENT_CYAN
    p2.space_after = Pt(36)
    
    p3 = tf1.add_paragraph()
    p3.text = "Presented by:\nCH.SC.U4CSE24127\nCH.SC.U4CSE24121"
    p3.font.size = Pt(16)
    p3.font.color.rgb = TEXT_MUTED

    # =========================================================
    # SLIDE 2: Problem Statement
    # =========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_bg(slide2)
    
    # Title
    tbox = slide2.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Problem Statement"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    # Card 1: What is the problem?
    add_card(slide2, Inches(1.0), Inches(1.4), Inches(5.4), Inches(3.6), "What is the problem?")
    tb = slide2.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(5.0), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Fraudulent transactions hide inside millions of genuine ones",
        "Fraud patterns constantly evolve and adapt",
        "Severe class imbalance: only ~0.1%–7% transactions are fraud",
        "High dimensionality and non-linear feature correlations"
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    # Card 2: Why solve it?
    add_card(slide2, Inches(6.9), Inches(1.4), Inches(5.4), Inches(3.6), "Why solve it?")
    tb = slide2.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.0), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Card, UPI, and online fraud cause billions in losses",
        "Fraud must be flagged in milliseconds before settlement",
        "Classical linear models struggle with non-linear feature overlaps",
        "Reduces costly false-positive alarms for financial institutions"
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    # Card 3: Why does it matter?
    add_card(slide2, Inches(1.0), Inches(5.3), Inches(11.3), Inches(1.7), "Why does it matter?")
    tb = slide2.shapes.add_textbox(Inches(1.2), Inches(5.9), Inches(10.9), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Protects customer trust and prevents unauthorized account takeovers.\n• Demonstrates how Quantum Machine Learning (QML) operates in high-stakes, real-world finance."
    p.font.size = Pt(15)
    p.font.color.rgb = TEXT_MUTED

    # =========================================================
    # SLIDE 3: Objectives & Approach
    # =========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_bg(slide3)
    
    tbox = slide3.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Objectives & Technical Approach"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide3, Inches(1.0), Inches(1.4), Inches(5.4), Inches(3.2), "How we solve it")
    tb = slide3.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(5.0), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Build a classifier that reliably labels transactions as Fraud (1) or Legit (0)",
        "Map classical transaction data into a high-dimensional Hilbert space",
        "Exploit quantum state overlap to separate complex decision boundaries"
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    add_card(slide3, Inches(6.9), Inches(1.4), Inches(5.4), Inches(3.2), "Our Approach")
    tb = slide3.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.0), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Hybrid Classical-Quantum pipeline architecture",
        "Classical: cleaning, scaling, and feature selection (4 key features)",
        "Quantum: ZZ Feature Map encoding + Fidelity Quantum Kernel matrix"
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    add_card(slide3, Inches(1.0), Inches(4.9), Inches(11.3), Inches(2.1), "Algorithm Used")
    tb = slide3.shapes.add_textbox(Inches(1.2), Inches(5.6), Inches(10.9), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Quantum Support Vector Machine (QSVM / QSVC)\n• Quantum Kernel: FidelityQuantumKernel with ZZFeatureMap (feature_dimension=4, reps=2)\n• Hardware / Backend: Local Quantum Statevector Simulator (Qiskit 2.x + Qiskit Aer)"
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_CYAN

    # =========================================================
    # SLIDE 4: What is QSVM?
    # =========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_bg(slide4)
    
    tbox = slide4.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "What is QSVM?"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide4, Inches(1.0), Inches(1.4), Inches(5.4), Inches(5.5), "Concept")
    tb = slide4.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(5.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    paragraphs = [
        "A classical SVM finds the optimal hyper-plane separating two classes in a transformed feature space.",
        "QSVM replaces the classical mathematical kernel (RBF / Polynomial) with a Quantum Kernel evaluated via a quantum circuit.",
        "Transactions are mapped into a 16-dimensional quantum Hilbert space (4 qubits) using non-linear ZZ entanglement.",
        "Quantum Kernel Similarity:\nK(xi, xj) = |<phi(xj)|phi(xi)>|^2\nmeasures transition fidelity between quantum states."
    ]
    for para in paragraphs:
        p = tf.add_paragraph()
        p.text = para
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    add_card(slide4, Inches(6.9), Inches(1.4), Inches(5.4), Inches(5.5), "7-Step Pipeline")
    tb = slide4.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    steps = [
        "1. Ingest, clean, and scale financial transaction data.",
        "2. Select / reduce to 4 core features for 4-qubit register.",
        "3. Stratify and balance fraud vs. legitimate training batch.",
        "4. Encode transactions using ZZ Feature Map (reps=2).",
        "5. Execute quantum circuits to compute Kernel Matrix K.",
        "6. Train classical Support Vector Classifier on matrix K.",
        "7. Classify real-time incoming transactions: Fraud or Legit."
    ]
    for step in steps:
        p = tf.add_paragraph()
        p.text = step
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    # =========================================================
    # SLIDE 5: Proposed Algorithm: Pseudocode and Flow
    # =========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_bg(slide5)
    
    tbox = slide5.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Proposed Algorithm: Pseudocode & Flow"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide5, Inches(1.0), Inches(1.4), Inches(6.8), Inches(5.5), "Algorithm Pseudocode")
    tb = slide5.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(6.4), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    code_text = (
        "INPUT: Transactions D, Labels y in {0, 1}\n\n"
        "1. X_scaled <- Preprocess_and_Scale(D)\n"
        "   X_q <- Select_Features(X_scaled, 4 features)\n\n"
        "2. Split (X_train, y_train), (X_test, y_test)\n\n"
        "3. FOR each pair (x_i, x_j) in X_train:\n"
        "       Construct circuit U_phi(x_j)^dagger * U_phi(x_i)\n"
        "       Measure ground state probability |0000>:\n"
        "       K[i, j] <- |<phi(x_j)|phi(x_i)>|^2\n\n"
        "4. Train Dual SVM Classifier on kernel matrix K:\n"
        "       alpha* <- argmin W(alpha) subject to constraints\n\n"
        "5. FOR new transaction t:\n"
        "       decision <- SUM(alpha_i * y_i * K(x_i, t)) + b\n"
        "       Fraud_Risk_Score <- Sigmoid(decision)\n"
        "       IF Fraud_Risk_Score >= Threshold: FLAG FRAUD\n"
        "       ELSE: APPROVE TRANSACTION"
    )
    p = tf.paragraphs[0]
    p.text = code_text
    p.font.size = Pt(13)
    p.font.name = "Consolas"
    p.font.color.rgb = ACCENT_CYAN

    add_card(slide5, Inches(8.1), Inches(1.4), Inches(4.2), Inches(5.5), "System Flow")
    tb = slide5.shapes.add_textbox(Inches(8.3), Inches(2.2), Inches(3.8), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    flow_steps = [
        "Incoming Transaction",
        "        ↓",
        "Feature Preprocessing",
        "        ↓",
        "4 Selected Features",
        "        ↓",
        "ZZ Quantum Feature Map",
        "        ↓",
        "Fidelity Quantum Kernel",
        "        ↓",
        "QSVC Classification",
        "        ↓",
        "Real-Time Fraud Alert"
    ]
    for s in flow_steps:
        p = tf.add_paragraph()
        p.text = s
        p.font.size = Pt(14)
        p.font.bold = True if "↓" not in s else False
        p.font.color.rgb = TEXT_WHITE if "↓" not in s else ACCENT_PURPLE

    # =========================================================
    # SLIDE 6: Algorithm Complexity
    # =========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_bg(slide6)
    
    tbox = slide6.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Algorithm Complexity Analysis"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide6, Inches(1.0), Inches(1.4), Inches(3.6), Inches(3.4), "Time: Training")
    tb = slide6.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(3.2), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Kernel Matrix: O(N² · shots) circuit evaluations.\n\n• SVM Optimization: O(N²) to O(N³) quadratic programming solve."
    p.font.size = Pt(15)
    p.font.color.rgb = TEXT_MUTED

    add_card(slide6, Inches(4.8), Inches(1.4), Inches(3.6), Inches(3.4), "Time: Prediction")
    tb = slide6.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.2), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• O(M · shots) kernel evaluations per incoming transaction.\n\n• M = number of support vectors (M ≤ N)."
    p.font.size = Pt(15)
    p.font.color.rgb = TEXT_MUTED

    add_card(slide6, Inches(8.6), Inches(1.4), Inches(3.6), Inches(3.4), "Space Complexity")
    tb = slide6.shapes.add_textbox(Inches(8.8), Inches(2.2), Inches(3.2), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• O(N²) memory for storing the Gram matrix.\n\n• Only n = 4 qubits (16-dimensional quantum state space)."
    p.font.size = Pt(15)
    p.font.color.rgb = TEXT_MUTED

    add_card(slide6, Inches(1.0), Inches(5.1), Inches(11.2), Inches(1.9), "Why It Matters (Practical NISQ Considerations)")
    tb = slide6.shapes.add_textbox(Inches(1.2), Inches(5.7), Inches(10.8), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• N = training samples. The O(N²) kernel calculation is the main computational bottleneck, which is why we train on a representative subset.\n• Circuit depth remains shallow (2 repetitions), suitable for current NISQ devices without decoherence breakdown.\n• Honest academic note: No exponential quantum speedup is claimed over classical SVM; the key benefit is accessing a richer quantum feature Hilbert space."
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_MUTED

    # =========================================================
    # SLIDE 7: Implementation & Tools
    # =========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_bg(slide7)
    
    tbox = slide7.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Implementation & Technology Stack"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide7, Inches(1.0), Inches(1.4), Inches(4.8), Inches(5.5), "Tools Used")
    tb = slide7.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(4.4), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tools = [
        "• Language: Python 3.13 / 3.11+",
        "• Qiskit (2.4.2): Quantum circuits & simulation",
        "• Qiskit Aer (0.17.2): High-performance local quantum simulator",
        "• qiskit-machine-learning (0.9.1): QSVC & FidelityQuantumKernel",
        "• scikit-learn (1.9.0): Preprocessing & Classical SVM baseline",
        "• Streamlit (1.65.0): Real-time interactive demonstration dashboard",
        "• Dataset: 5,000 Financial Transactions (Academic Synthetic Dataset)"
    ]
    for t in tools:
        p = tf.add_paragraph()
        p.text = t
        p.font.size = Pt(15)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    add_card(slide7, Inches(6.1), Inches(1.4), Inches(6.2), Inches(5.5), "Qiskit Machine Learning Code")
    tb = slide7.shapes.add_textbox(Inches(6.3), Inches(2.2), Inches(5.8), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    impl_code = (
        "# 1. Define 4-qubit ZZ Feature Map\n"
        "fmap = ZZFeatureMap(feature_dimension=4, reps=2)\n\n"
        "# 2. Build Quantum Fidelity Kernel\n"
        "qk = FidelityQuantumKernel(feature_map=fmap)\n\n"
        "# 3. Instantiate Quantum Support Vector Classifier\n"
        "qsvc = QSVC(quantum_kernel=qk)\n\n"
        "# 4. Fit on preprocessed training transactions\n"
        "qsvc.fit(X_train, y_train)\n\n"
        "# 5. Predict on unseen test transactions\n"
        "y_pred = qsvc.predict(X_test)\n\n"
        "# 6. Generate real-time Fraud Risk Score\n"
        "risk_score = qsvc.decision_function(new_txn)"
    )
    p = tf.paragraphs[0]
    p.text = impl_code
    p.font.size = Pt(14)
    p.font.name = "Consolas"
    p.font.color.rgb = ACCENT_CYAN

    # =========================================================
    # SLIDE 8: Results and Analysis (FILLED IN!)
    # =========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_bg(slide8)
    
    tbox = slide8.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Results and Analysis"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    # Table comparing QSVM vs Classical SVM
    rows, cols = 6, 3
    table_shape = slide8.shapes.add_table(rows, cols, Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.6))
    table = table_shape.table

    table.columns[0].width = Inches(3.7)
    table.columns[1].width = Inches(3.8)
    table.columns[2].width = Inches(3.8)

    headers = ["Metric", "Quantum QSVM (4-Qubit)", "Classical SVM Baseline"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(67, 40, 115)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(16)
            p.font.color.rgb = ACCENT_CYAN

    # Real experimental comparison data
    metrics_data = [
        ("Accuracy", "89.67%", "91.80%"),
        ("Precision", "28.57%", "44.12%"),
        ("Recall", "14.28%", "64.29%"),
        ("F1-Score", "0.1905", "0.5263"),
        ("ROC-AUC", "0.5630", "0.9124")
    ]

    for row_idx, (m, q_val, c_val) in enumerate(metrics_data, start=1):
        table.cell(row_idx, 0).text = m
        table.cell(row_idx, 1).text = q_val
        table.cell(row_idx, 2).text = c_val
        for c_idx in range(3):
            cell = table.cell(row_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(15)
                p.font.color.rgb = TEXT_WHITE

    add_card(slide8, Inches(1.0), Inches(4.5), Inches(11.3), Inches(2.5), "Observations & Academic Insights")
    tb = slide8.shapes.add_textbox(Inches(1.2), Inches(5.1), Inches(10.9), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    obs = [
        "• Class Imbalance Priority: Because fraud accounts for only ~7% of data, Recall and F1-score are far more informative than raw Accuracy.",
        "• Classical SVM Strength: Classical RBF SVM trained on all features achieves stronger recall (~64.3%) and higher discrimination (0.912 ROC-AUC).",
        "• Quantum QSVM Validity: QSVM with ZZFeatureMap successfully maps non-linear correlations into 4-qubit Hilbert space and predicts unseen transactions.",
        "• Quantum Advantage Context: Current NISQ simulators do not exhibit quantum supremacy over mature classical SVMs, but prove the feasibility of the QML pipeline."
    ]
    for o in obs:
        p = tf.add_paragraph()
        p.text = o
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    # =========================================================
    # SLIDE 9: Conclusion
    # =========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_bg(slide9)
    
    tbox = slide9.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.3), Inches(0.8))
    p = tbox.text_frame.paragraphs[0]
    p.text = "Conclusion & Future Work"
    p.font.bold = True
    p.font.size = Pt(32)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide9, Inches(1.0), Inches(1.4), Inches(5.4), Inches(2.8), "What We Achieved")
    tb = slide9.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(5.0), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Designed and tested a hybrid quantum-classical fraud detection pipeline.",
        "Built a genuine 4-qubit QSVM with ZZFeatureMap and Fidelity Quantum Kernel.",
        "Created an interactive Streamlit live demo with real-time scoring & transaction audit history."
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(6)

    add_card(slide9, Inches(6.9), Inches(1.4), Inches(5.4), Inches(2.8), "Problem Addressed?")
    tb = slide9.shapes.add_textbox(Inches(7.1), Inches(2.2), Inches(5.0), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Yes: Incoming transactions are classified into Legitimate or Suspicious in milliseconds.",
        "Fraud risk scores dynamically inform approval or manual verification decisions.",
        "Handles extreme non-linear transactions without crashing."
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(6)

    add_card(slide9, Inches(1.0), Inches(4.5), Inches(5.4), Inches(2.5), "Key Observations")
    tb = slide9.shapes.add_textbox(Inches(1.2), Inches(5.1), Inches(5.0), Inches(1.7))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Quantum kernels provide access to a rich exponential Hilbert space.",
        "Requires balanced, reduced datasets on NISQ devices to maintain reasonable simulation times."
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(6)

    add_card(slide9, Inches(6.9), Inches(4.5), Inches(5.4), Inches(2.5), "Future Work")
    tb = slide9.shapes.add_textbox(Inches(7.1), Inches(5.1), Inches(5.0), Inches(1.7))
    tf = tb.text_frame
    tf.word_wrap = True
    for item in [
        "Deployment onto physical cryogenic QPUs (IBM Quantum Heron / Eagle).",
        "Quantum error mitigation (Zero-Noise Extrapolation).",
        "Streaming Kafka ingestion for real-time banking gateways."
    ]:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(6)

    # =========================================================
    # SLIDE 10: Thank You
    # =========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_bg(slide10)
    
    tb10 = slide10.shapes.add_textbox(Inches(2.0), Inches(2.5), Inches(9.3), Inches(2.5))
    tf10 = tb10.text_frame
    p = tf10.paragraphs[0]
    p.text = "Thank You"
    p.font.bold = True
    p.font.size = Pt(60)
    p.font.color.rgb = TEXT_WHITE
    p.alignment = PP_ALIGN.CENTER
    
    p2 = tf10.add_paragraph()
    p2.text = "Questions & Answers | Live Demonstration"
    p2.font.size = Pt(22)
    p2.font.color.rgb = ACCENT_CYAN
    p2.alignment = PP_ALIGN.CENTER

    prs.save(output_pptx)
    print(f"Presentation saved successfully to: {output_pptx}")
    return output_pptx

if __name__ == "__main__":
    create_deck()
