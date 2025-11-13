## 🧠 Review Sentiment Classification

### 📘 Project Description

This project focuses on **sentiment classification of product reviews** using both **traditional machine learning** and **transformer-based models**.
We compare the performance of a **TF–IDF + Logistic Regression** baseline against a **fine-tuned Transformer** model on a large dataset of labeled reviews (positive / neutral / negative).

The final transformer model achieved **84% accuracy** and strong macro F1 performance, significantly outperforming the baseline.

---

### 🚀 Key Features

* End-to-end sentiment analysis pipeline
* Preprocessing, feature extraction, and evaluation modules
* Baseline (TF–IDF + Logistic Regression) and Transformer comparison
* Modular architecture for reproducibility and extension
* Experiment tracking and model saving

---

### 📂 Project Structure

```
REVIEW_CLASSIFICATION/
├── data/
│   ├── raw/                     # Original raw data
│   └── processed/               # Cleaned & preprocessed CSV files
│       ├── train.csv
│       └── test.csv
│
├── experiments/                 # Experiment results & logs
│
├── models/                      # Saved models and training history
│   ├── baseline.joblib
│   ├── best_model_epoch3_f10.8012
│   └── training_history.csv
│
├── notebooks/
│   └── sentiment-pipeline.ipynb # Jupyter notebook with full pipeline
│
├── src/                         # Core source code
│   ├── baseline.py              # TF–IDF + Logistic Regression
│   ├── data_prep.py             # Data loading and preprocessing
│   ├── eval.py                  # Evaluation and reporting
│   ├── features.py              # Feature extraction
│   ├── transformer.py           # Transformer fine-tuning
│   └── utils.py                 # Helper functions
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

### ⚙️ Architecture Overview

```text
        ┌─────────────────────────────┐
        │        Raw Reviews          │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Data Preprocessing    │  →  Cleaning, tokenization, labeling
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Feature Engineering    │  →  TF–IDF or Transformer embeddings
        └────────────┬────────────┘
                     │
        ┌─────┬──────▼──────┬─────┐
        │LogReg│Transformer │Other│
        └─────┴──────┬──────┴─────┘
                     │
        ┌────────────▼────────────┐
        │        Evaluation       │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Metrics & Comparison  │
        └─────────────────────────┘
```

---

### 📊 Model Performance

#### 🔹 **Transformer Fine-tuned Model**

| Metric            |   Value   |
| :---------------- | :-------: |
| Accuracy          | **0.840** |
| F1 (macro)        | **0.801** |
| Precision (macro) |   0.804   |
| Recall (macro)    |   0.799   |

**Classification Report (Validation):**

```
              precision    recall  f1-score   support
negative       0.89        0.90       0.89     20000
neutral        0.64        0.59       0.61     10000
positive       0.89        0.91       0.90     20000
accuracy                               0.84     50000
```

✅ Saved best model to: `models/best_model_epoch3_f10.8012`

---

#### 🔸 **Baseline: TF–IDF + Logistic Regression**

| Metric     |   Value   |
| :--------- | :-------: |
| Accuracy   | **0.800** |
| F1 (macro) | **0.770** |

**Classification Report:**

```
              precision    recall  f1-score   support
negative       0.87        0.85       0.86     26000
neutral        0.55        0.61       0.58     13000
positive       0.88        0.85       0.86     26000
```

---

### 💾 Installation

```bash
git clone https://github.com/weaknessofuniverse/review-classification.git
cd review-classification
pip install -r requirements.txt
```

---

### 🧩 Usage

Run preprocessing:

```bash
python src/data_prep.py
```

Train baseline model:

```bash
python src/baseline.py
```

Fine-tune transformer:

```bash
python src/transformer.py
```

Evaluate results:

```bash
python src/eval.py
```

---

### 📈 Future Work

* Expand dataset with multilingual reviews
* Experiment with LLaMA or Mistral-based encoders
* Integrate real-time inference API

