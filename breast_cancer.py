import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_SIZE = (300, 300)
train_dir = "datasett/train"
val_dir = "datasett/val"
save_path = "breast_cancer_efficientnetv2.keras" # Format moderne

# --- 1. GÉNÉRATEURS DE DONNÉES ---
# IMPORTANT : Pas de rescale=1./255 avec EfficientNetV2 + preprocess_input
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,      # Utile pour les biopsies
    fill_mode='reflect'
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',     # Mode binaire (0/1)
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 2. GESTION DU DÉSÉQUILIBRE (Class Weights) ---
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(zip(np.unique(labels), class_weights))
print(f"⚖️ Poids des classes calculés : {class_weight_dict}")

# --- 3. MODÈLE (Approche directe) ---
base_model = EfficientNetV2B3(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)

# On rend le modèle trainable, MAIS on gèle tout sauf les 50 dernières couches
base_model.trainable = True
for layer in base_model.layers[:-50]: 
    layer.trainable = False

# Architecture de la tête (Classification)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Stabilise les poids venant d'ImageNet
x = Dropout(0.4)(x)         # Anti-overfitting puissant
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- 4. COMPILATION ---
# On utilise un Learning Rate faible (1e-4) car on touche aux couches du CNN
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# --- 5. CALLBACKS ---
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
    ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# --- 6. ENTRAÎNEMENT UNIQUE ---
print("\n🚀 Lancement de l'entraînement (Fine-tuning direct des 50 dernières couches)...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=callbacks,
    class_weight=class_weight_dict, # Crucial pour l'équilibre
    verbose=1
)

print(f"\n✅ Modèle terminé et sauvegardé sous : {save_path}")