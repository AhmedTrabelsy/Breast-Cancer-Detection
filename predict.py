import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Vérifier que le modèle existe
if not os.path.exists("colon_cancer_final.h5"):
    print("❌ Erreur : Le fichier breast_cancer_model_final.h5 n'existe pas ! Entraîne d'abord le modèle.")
    exit()

model = load_model("colon_cancer_final.h5")

# Chemin de l'image
img_path = "test.png"  # Change ici

if not os.path.exists(img_path):
    print(f"❌ Erreur : L'image {img_path} n'existe pas !")
    exit()

img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img) / 255.
img_array = np.expand_dims(img_array, axis=0)

# Prédiction corrigée : récupération des probabilités pour les 2 classes
pred_probs = model.predict(img_array)[0]
prob_malignant = pred_probs[1]  # Classe 1 = malignant (selon l'ordre alphabétique des dossiers)
predicted_class = np.argmax(pred_probs)  # 0 = benign, 1 = malignant

print(f"Probabilités : Benign = {pred_probs[0]:.4f}, Malignant = {prob_malignant:.4f}")
print(f"Score brut (malignant) : {prob_malignant:.4f} (plus proche de 1 = cancer, plus proche de 0 = normal)")

if predicted_class == 1:
    print("🔴 Prédiction : Cancer du côlon (colon_aca)")
else:
    print("🟢 Prédiction : Normal (colon_n)")