import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

df = pd.read_csv("results_with_pca_clusters.csv")

# ============================================================
# 2. SUPPRESSION DES COLONNES NON PERTINENTES / FUITES
# ============================================================

target_col = "Risk_Level"

# Colonnes qui causent une fuite (contiennent le risque déjà calculé)
leak_cols = [c for c in df.columns if "score" in c.lower() or "overall_risk" in c.lower()]

# Colonnes inutiles
drop_base = ["Patient_ID", "Cluster", "Recommendations", "Cancer_Type", target_col]

drop_cols = list(set(drop_base + leak_cols))

print("Colonnes supprimées (pour éviter fuites et bruit) :")
print(drop_cols)

# ============================================================
# 3. SÉLECTION DES FEATURES
# ============================================================

# On enlève aussi les colonnes PCA, car elles ne sont pas utiles ici
feature_cols = [c for c in df.columns if c not in drop_cols and "PC" not in c]

X = df[feature_cols]
y = df[target_col].astype(str)

# ============================================================
# 4. ENCODAGE
# ============================================================

# Encode la cible
le = LabelEncoder()
y_enc = le.fit_transform(y)

# One-hot encode des colonnes catégorielles
X_enc = pd.get_dummies(X, drop_first=True)

# ============================================================
# 5. TRAIN / TEST SPLIT AVEC STRATIFY
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ============================================================
# 6. NORMALISATION (seulement pour KNN)
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. MODÈLES : KNN & DECISION TREE
# ============================================================

# ---- KNN ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

# ---- Decision Tree ----
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)  # pas besoin de scaling
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# ============================================================
# 8. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n====================")
print("     RÉSULTATS")
print("====================")
print(f"Accuracy KNN             : {acc_knn:.4f}")
print(f"Accuracy Decision Tree   : {acc_dt:.4f}")

print("\n--- Classification Report KNN ---")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))

print("\n--- Classification Report Decision Tree ---")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

# ============================================================
# 9. MATRICES DE CONFUSION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_knn),
            annot=True, cmap="Blues", fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0])
axes[0].set_title(f"Confusion Matrix - KNN (Acc : {acc_knn:.2f})")
axes[0].set_xlabel("Prédit")
axes[0].set_ylabel("Vrai")

sns.heatmap(confusion_matrix(y_test, y_pred_dt),
            annot=True, cmap="Greens", fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1])
axes[1].set_title(f"Confusion Matrix - Decision Tree (Acc : {acc_dt:.2f})")
axes[1].set_xlabel("Prédit")
axes[1].set_ylabel("Vrai")

plt.tight_layout()
plt.show()

# ============================================================
# 10. GRAPHIQUE COMPARATIF
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(["KNN", "Decision Tree"], [acc_knn, acc_dt],
        color=["blue", "green"], alpha=0.8)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Comparaison des modèles")

for i, v in enumerate([acc_knn, acc_dt]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

plt.show()
