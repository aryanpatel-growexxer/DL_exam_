# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:**
**Student ID:**
**Submission date:**

---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

---

## My model

**Architecture:**
<!-- Describe your network: layer sizes, activations, regularisation -->
It consits of 2 hidden layer
Input layer : 14 features
Hidden first : 64 neurons 
Hidden second : 32 neurons 
Output layer : 1 layer 
Their is a Batchnorm in each layer.

**Key preprocessing decisions:**
<!-- Summarise the most important choices — 2–3 sentences -->
I have removed the date , day of week.
I have encoded categorical data
I have imputed age and glucose data.

**How I handled class imbalance:**
<!-- What technique and why -->
I have upsampled to train the model to handle with class imbalance.

---

## Results on validation set

| Metric | Value |
|--------|-------|
| AUROC | 0.65 |
| F1 (minority class) | 0.62 |
| Precision (minority) | 0.61 |
| Recall (minority) | 0.64 |
| Decision threshold used | 0.7 |

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — pretrained weights included)

```bash
python notebooks/solution.ipynb  # or run cells in order
```

### 3. Run inference on the test set

```bash
python src/predict.py --input data/test.csv --output predictions.csv
```

The output CSV will contain two columns: `patient_id` and `readmission_probability`.

---

## Repository structure

```
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md
```

---

## Limitations and honest assessment

<!-- What would you improve with more time? Where might this model fail in production? -->
