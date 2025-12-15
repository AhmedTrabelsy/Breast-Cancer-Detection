import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # IMPORTANT
import os

# -----------------------------
# Chargement du modèle
# -----------------------------
model_path = "colon_cancer_model.keras"

if not os.path.exists(model_path):
    print("❌ Erreur : Le modèle n'existe pas.")
    exit()

model = load_model(model_path)
print("✅ Modèle chargé.")

# -----------------------------
# Image à tester
# -----------------------------
img_path = "test.jpg"  # Vérifiez que l'image existe

if not os.path.exists(img_path):
    print(f"❌ Erreur : L'image '{img_path}' est introuvable.")
    exit()

# --- PRÉTRAITEMENT CORRIGÉ ---
# 1. Charger l'image
img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img)

# 2. Ajouter la dimension du batch (1, 300, 300, 3)
img_array = np.expand_dims(img_array, axis=0)

# 3. Utiliser la fonction native d'EfficientNet (PAS de division par 255)
img_array = preprocess_input(img_array)

# -----------------------------
# Prédiction et Interprétation
# -----------------------------
# La sortie (pred) est la probabilité de la classe 1.
# Par ordre alphabétique : 0 = colon_aca (Cancer), 1 = colon_n (Normal)
prob_normal = model.predict(img_array)[0][0]
prob_cancer = 1.0 - prob_normal

print(f"\n📌 Image testée : {img_path}")
print(f"Probabilité Cancer (colon_aca) : {prob_cancer:.4f}")
print(f"Probabilité Normal (colon_n)   : {prob_normal:.4f}")

# Seuil de décision (ajustable)
threshold = 0.5 

if prob_cancer > threshold:
    result = "🔴 DANGER : Cancer détecté (colon_aca)"
    confidence = prob_cancer
else:
    result = "🟢 RAS : Tissu normal (colon_n)"
    confidence = prob_normal

print(f"Résultat : {result}")
print(f"Confiance : {confidence:.1%}")