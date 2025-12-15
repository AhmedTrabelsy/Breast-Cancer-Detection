import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # IMPORTANT
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# --- CONFIGURATION ---
BATCH_SIZE = 16 # Essayez 32 si votre GPU le permet
IMG_SIZE = (300, 300)
train_dir = "dataset/train"
val_dir = "dataset/val"

# --- 1. GÉNÉRATEURS (CORRIGÉS) ---
# Note : On retire rescale=1./255 et on utilise preprocess_input
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Normalisation native EfficientNet
    rotation_range=20,      # Réduit un peu pour ne pas trop déformer les tissus
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,     # Ajouté : en histologie, le haut/bas n'a pas de sens strict
    fill_mode='reflect'     # Meilleur que 'nearest' pour les textures continues
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 2. GESTION DU DÉSÉQUILIBRE ---
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(zip(np.unique(labels), class_weights))
print(f"⚖️ Poids des classes : {class_weight_dict}")

# --- 3. MODÈLE (OPTIMISÉ) ---
base_model = EfficientNetV2B3(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)

# On débloque directement quelques couches du haut car ImageNet est trop différent du médical
base_model.trainable = True
for layer in base_model.layers[:-50]: # On gèle tout sauf les 50 dernières couches
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Stabilise l'apprentissage
x = Dropout(0.4)(x)         # Réduit l'overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- 4. COMPILATION AVEC LEARNING RATE ADAPTATIF ---
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # LR plus faible pour commencer

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# --- 5. CALLBACKS ---
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint("colon_cancer_model.keras", save_best_only=True, monitor='val_accuracy'), # Format .keras recommandé
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1) # Réduit le LR si ça stagne
]

# --- 6. ENTRAÎNEMENT ---
print("\n🚀 Début de l'entraînement...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40, # On augmente car le LR est plus bas
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n✅ Entraînement terminé.")