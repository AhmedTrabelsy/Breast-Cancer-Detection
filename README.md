# 🧠 Breast Cancer Detection using Machine Learning

A complete reproduction and extension of the research paper:

**"On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset"** by Abien Fred M. Agarap.

This project implements, evaluates, and compares several machine learning algorithms on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, following the same methodology, preprocessing, and hyperparameters used in the paper.

---

## 📌 Project Overview

The goal of this project is to benchmark multiple ML algorithms on a binary breast cancer classification problem (Benign vs Malignant), using:

- GRU-SVM
- Linear Regression (as classifier)
- Multilayer Perceptron (MLP) — _fully reproduced from the paper_
- Nearest Neighbor
- Softmax Regression
- L2-SVM

The implementation includes:

- Standardization of the dataset
- 70/30 train/test split
- Reproduction of hyperparameters from the article
- Detailed performance metrics for each model
- Confusion matrix and sensitivity/specificity analysis

---

## 📂 Project Structure

```
📦 project-root
│
├── EVALUATION.ipynb         # Main notebook with all ML model implementations
├── donnees_nettoyees.xlsx    # Cleaned dataset used for experiments
├── article.pdf                # Reference research article (optional)
└── README.md                 # You are here
```

---

## ⚙️ Implemented Algorithms

### 1️⃣ GRU-SVM

Hybrid recurrent model combining:

- **GRU** for sequence-like representation
- **L2-SVM** as final classifier

Training uses:

- Adam optimizer
- Batch size = 128
- Epochs = 3000

---

### 2️⃣ Linear Regression (as classifier)

Although unconventional, the paper applies linear regression with:

- MSE loss
- SGD optimizer
- Thresholding to convert regression scores into binary classes

---

### 3️⃣ Multilayer Perceptron (MLP)

Fully reproduced according to the paper:

- Architecture: **500 – 500 – 500**
- Activation: **ReLU**
- Optimizer: **SGD (lr = 0.01)**
- Loss: **binary crossentropy**
- Epochs: **3000**
- Batch size: **128**

This model provides the **best accuracy (~99%)** in the paper.

---

### 4️⃣ Nearest Neighbor

Distance-based model using:

- L1 (Manhattan)
- L2 (Euclidean)

No training phase required.

---

### 5️⃣ Softmax Regression

Multinomial logistic regression adapted to binary classification.
Uses:

- Cross-entropy loss
- SGD optimizer

---

### 6️⃣ L2-SVM

Standard linear Support Vector Machine with:

- L2 regularization
- Differentiable loss
- Adam optimizer

---

## 📊 Evaluation Metrics

Each model is evaluated using the project's `evaluate()` function, printing:

- **Accuracy**
- **Sensitivity (TPR)**
- **Specificity (TNR)**
- **False Positive Rate (FPR)**
- **False Negative Rate (FNR)**
- **Confusion Matrix**
- **Epoch count (if applicable)**

Format example:

```
══════════════════════════════════════════
📌 Model : MLP
══════════════════════════════════════════
Accuracy : 99.041534%
TPR (Sensitivity) : ...
TNR (Specificity) : ...
FPR : ...
FNR : ...
Confusion matrix:
[[TN FP]
 [FN TP]]
══════════════════════════════════════════
```

---

## 🔧 Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** (MLP, GRU-SVM)
- **scikit-learn** (SVM, Linear Regression, Softmax, KNN)
- **NumPy & Pandas** (data processing)
- **Matplotlib** (visualizations)

---

## 🚀 How to Run the Notebook

1. Install dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

2. Open the notebook:

```bash
jupyter notebook EVALUATION.ipynb
```

3. Run all cells.

---

## 📈 Results Summary

All algorithms achieve above **90% accuracy**, with MLP and L2-SVM performing the best, consistent with the research paper.

You can find detailed numerical results inside the notebook.

---

## 🧪 Future Improvements

Possible extensions:

- Add k-fold cross-validation (recommended by the article)
- Add hyperparameter tuning (GridSearch, RandomSearch)
- Compare with modern models (XGBoost, Random Forest, Deep CNN)
- Export models for deployment

---

## 🔗 Useful Links

- 📄 **Research Paper:** [https://sci-hub.se/10.1145/3184066.3184080](https://sci-hub.se/10.1145/3184066.3184080)
- 📊 **WDBC Dataset (Kaggle):** [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)

## 📜 Reference

Agarap, A. F. (2018). _On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset_. ICMLSC.

---
