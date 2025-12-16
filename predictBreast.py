import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # IMPORTANT
import os

# -----------------------------
# Chargement du modèle
# -----------------------------
model_path = "breast_cancer_efficientnetv2.keras"  # Nom du modèle sauvegardé

if not os.path.exists(model_path):
    print("❌ Erreur : Le modèle n'existe pas à l'emplacement spécifié.")
    exit()

model = load_model(model_path)
print("✅ Modèle chargé avec succès.")

# -----------------------------
# Image à tester
# -----------------------------
img_path = "test.jpg"  # Change ce chemin vers ton image de test

if not os.path.exists(img_path):
    print(f"❌ Erreur : L'image '{img_path}' est introuvable.")
    exit()

# --- PRÉTRAITEMENT (spécifique à EfficientNetV2) ---
img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Ajout batch dimension
img_array = preprocess_input(img_array)       # Pré-traitement EfficientNetV2

# -----------------------------
# Prédiction et Interprétation
# -----------------------------
prob_benign = model.predict(img_array)[0][0]      # Probabilité classe 1 = bénin
prob_malignant = 1.0 - prob_benign                # Probabilité cancer

print(f"\n📌 Image testée : {img_path}")
print(f"Probabilité Cancer       : {prob_malignant:.4f}")
print(f"Probabilité Non cancer   : {prob_benign:.4f}")

# Seuil de décision
threshold = 0.5

if prob_malignant > threshold:
    resultat_final = "Cancer"
    confidence = prob_malignant
    emoji = "🔴"
else:
    resultat_final = "Non cancer"
    confidence = prob_benign
    emoji = "🟢"

# Affichage final clair et direct
print(f"\n{emoji} RÉSULTAT FINAL : {resultat_final.upper()}")
print(f"Confiance du modèle : {confidence:.1%}")