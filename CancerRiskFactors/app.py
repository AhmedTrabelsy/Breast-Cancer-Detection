import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="Évaluation Risque Cancer", layout="centered")

# ============================================================
# FEATURES
# ============================================================
CANCER_TYPE_FEATURES = [
    'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity',
    'Family_History', 'Diet_Red_Meat', 'Diet_Salted_Processed',
    'Fruit_Veg_Intake', 'Physical_Activity', 'Air_Pollution',
    'Occupational_Hazards', 'BRCA_Mutation', 'H_Pylori_Infection',
    'Calcium_Intake', 'Overall_Risk_Score', 'BMI',
    'Physical_Activity_Level'
]

RECO_FEATURES = CANCER_TYPE_FEATURES + ['Cancer_Type']

# ============================================================
# CHARGEMENT MODÈLES QUESTIONNAIRE
# ============================================================
@st.cache_resource
def load_risk_model(): return joblib.load("model_reco.pkl")
@st.cache_resource
def load_cancer_model(): return joblib.load("model_cancer_type.pkl")
@st.cache_resource
def load_encoder(): return joblib.load("cancer_type_encoder.pkl")

risk_model = load_risk_model()
cancer_model = load_cancer_model()
cancer_type_encoder = load_encoder()

# ============================================================
# VALIDATION & RECOMMANDATIONS
# ============================================================
VALIDATION_RULES = {
    'Age': (0, 120), 'Smoking': (0, 10), 'Alcohol_Use': (0, 10), 'Obesity': (0, 10),
    'Diet_Red_Meat': (0, 10), 'Diet_Salted_Processed': (0, 10), 'Fruit_Veg_Intake': (0, 10),
    'Physical_Activity': (0, 10), 'Air_Pollution': (0, 10), 'Occupational_Hazards': (0, 10),
    'Calcium_Intake': (0, 10), 'Physical_Activity_Level': (0, 10), 'BMI': (10, 60),
    'Overall_Risk_Score': (0, 1), 'Gender': [0, 1], 'Family_History': [0, 1],
    'BRCA_Mutation': [0, 1], 'H_Pylori_Infection': [0, 1]
}

def validate_input(feature, value):
    try: value = float(value)
    except ValueError: return False, "⚠️ Valeur numérique requise."
    rule = VALIDATION_RULES.get(feature)
    if isinstance(rule, tuple):
        if not rule[0] <= value <= rule[1]:
            return False, f"⚠️ Plage : [{rule[0]} – {rule[1]}]."
    if isinstance(rule, list) and value not in rule:
        return False, f"⚠️ Valeurs possibles : {rule}"
    return True, value

def generate_recommendations(row, predicted_risk):
    recos = []
    def has(col): return col in row.index

    if has('Smoking') and row['Smoking'] >= 7: recos.append("Arrêter le tabac (aide tabacologue).")
    if has('Alcohol_Use') and row['Alcohol_Use'] >= 7: recos.append("Limiter l’alcool (≤1 verre/jour).")
    if has('Obesity') and row['Obesity'] >= 7: recos.append("Programme de perte de poids.")
    if has('BMI') and row['BMI'] >= 30: recos.append("Viser IMC < 25.")
    if has('Fruit_Veg_Intake') and row['Fruit_Veg_Intake'] <= 3: recos.append("≥5 portions fruits/légumes par jour.")
    if has('Diet_Red_Meat') and row['Diet_Red_Meat'] >= 7: recos.append("Réduire viande rouge.")
    if has('Diet_Salted_Processed') and row['Diet_Salted_Processed'] >= 7: recos.append("Limiter aliments transformés/salés.")
    if has('Physical_Activity') and row['Physical_Activity'] <= 3: recos.append("≥150 min activité modérée/semaine.")
    if has('Air_Pollution') and row['Air_Pollution'] >= 7: recos.append("Éviter zones polluées.")
    if has('Occupational_Hazards') and row['Occupational_Hazards'] >= 7: recos.append("Renforcer protections au travail.")
    if has('Family_History') and row['Family_History'] == 1: recos.append("Dépistage précoce recommandé.")
    if has('BRCA_Mutation') and row['BRCA_Mutation'] == 1: recos.append("Suivi génétique renforcé.")
    if has('H_Pylori_Infection') and row['H_Pylori_Infection'] == 1: recos.append("Traiter infection H. pylori.")

    risk_str = str(predicted_risk).lower()
    if "high" in risk_str: recos.append("🛑 **Risque ÉLEVÉ** → Consultation urgente.")
    elif "medium" in risk_str or "moderate" in risk_str: recos.append("⚠️ **Risque MOYEN** → Appliquer changements.")
    else: recos.append("✅ **Risque FAIBLE** → Maintenir mode de vie sain.")

    return list(dict.fromkeys(recos))

def recommend_for_patient(patient_features):
    df_cancer = pd.DataFrame([patient_features])[CANCER_TYPE_FEATURES]
    pred_encoded = cancer_model.predict(df_cancer)[0]
    cancer_type = cancer_type_encoder.inverse_transform([pred_encoded])[0]
    patient_features['Cancer_Type'] = cancer_type
    df_risk = pd.DataFrame([patient_features])[RECO_FEATURES]
    risk_pred = risk_model.predict(df_risk)[0]
    recos = generate_recommendations(df_risk.iloc[0], risk_pred)
    return cancer_type, risk_pred, recos

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ["Informations Générales", "Détails du Modèle", "Chatbot Prédiction"])

if page == "Informations Générales":
    st.title("Évaluation des risques liés au cancer")
    st.markdown("Deux modes : questionnaire personnalisé ou analyse d’image médicale (poumon, sein, peau).")

elif page == "Détails du Modèle":
    st.title("Modèles utilisés")
    st.markdown("""
    - **Questionnaire** : Random Forest + AdaBoost + règles expertes
    - **Image** : Transfer Learning (Xception pour poumon, EfficientNet/ResNet pour peau/sein)
    """)

else:  # Chatbot Prédiction
    st.title("🩺 Chatbot – Évaluation personnalisée du risque de cancer")

    mode = st.radio("Choisissez le mode :", [
        "Questionnaire interactif (facteurs de risque)",
        "Analyse d'image médicale (détection de tumeur)"
    ])

    # ============================= MODE QUESTIONNAIRE =============================
    if mode == "Questionnaire interactif (facteurs de risque)":
        if "current_conv" not in st.session_state:
            st.session_state.current_conv = {"messages": [], "responses": {}, "question_index": 0, "completed": False}

        conv = st.session_state.current_conv
        questions = {
            'Age': "Quel est votre âge ?", 'Gender': "Sexe ? (0=Femme, 1=Homme)",
            'Smoking': "Tabagisme (0-10) ?", 'Alcohol_Use': "Alcool (0-10) ?", 'Obesity': "Obésité perçue (0-10) ?",
            'Family_History': "Antécédents familiaux ? (0=Non, 1=Oui)", 'Diet_Red_Meat': "Viande rouge (0-10) ?",
            'Diet_Salted_Processed': "Aliments transformés/salés (0-10) ?", 'Fruit_Veg_Intake': "Fruits/légumes (0-10) ?",
            'Physical_Activity': "Activité physique (0-10) ?", 'Air_Pollution': "Exposition pollution (0-10) ?",
            'Occupational_Hazards': "Risques pro (0-10) ?", 'BRCA_Mutation': "Mutation BRCA ? (0=Non, 1=Oui)",
            'H_Pylori_Infection': "H. pylori ? (0=Non, 1=Oui)", 'Calcium_Intake': "Apport calcium (0-10) ?",
            'Overall_Risk_Score': "Score risque global connu (0-1) ? Sinon 0.", 'BMI': "IMC ?", 
            'Physical_Activity_Level': "Niveau activité global (0-10) ?"
        }
        keys = list(questions.keys())

        for msg in conv["messages"]:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])

        if conv.get("completed", False):
            st.success("Évaluation terminée !")
            cancer_type, risk_pred, recos = conv["results"]
            st.info(f"**Type prédit :** {cancer_type}")
            st.warning(f"**Risque :** {risk_pred}")
            st.subheader("Recommandations")
            for r in recos: st.markdown(f"• {r}")
            if st.button("Nouvelle évaluation"): 
                st.session_state.current_conv = {"messages": [{"role": "assistant", "content": "Nouvelle évaluation !"}], "responses": {}, "question_index": 0, "completed": False}
                st.rerun()

        else:
            if conv["question_index"] < len(keys):
                key = keys[conv["question_index"]]
                q = questions[key]
                if not conv["messages"] or conv["messages"][-1]["content"] != q:
                    with st.chat_message("assistant"): st.markdown(q)
                    conv["messages"].append({"role": "assistant", "content": q})

                if user_input := st.chat_input("Votre réponse..."):
                    valid, msg = validate_input(key, user_input)
                    if valid:
                        conv["responses"][key] = msg
                        with st.chat_message("user"): st.markdown(user_input)
                        conv["messages"].append({"role": "user", "content": user_input})
                        conv["question_index"] += 1
                        st.rerun()
                    else:
                        with st.chat_message("assistant"): st.error(msg)
                        conv["messages"].append({"role": "assistant", "content": msg})
            else:
                cancer_type, risk_pred, recos = recommend_for_patient(conv["responses"])
                conv["results"] = (cancer_type, risk_pred, recos)
                conv["completed"] = True
                with st.chat_message("assistant"):
                    st.markdown(f"**Type :** {cancer_type}\n**Risque :** {risk_pred}\n**Recommandations :**")
                    for r in recos: st.markdown(f"• {r}")
                conv["messages"].append({"role": "assistant", "content": f"Résultats : {cancer_type}, {risk_pred}\n" + "\n".join(f"• {r}" for r in recos)})
                st.rerun()

    # ============================= MODE IMAGE =============================
    else:
        st.markdown("### 🔬 Analyse d'une image médicale")
        st.warning("⚠️ **Outil éducatif uniquement** • Ne remplace PAS un diagnostic médical • Consultez un spécialiste.")

        cancer_type_selected = st.selectbox("Type d'image :", [
            "Lung (CT scan poumon)",
            "Breast (Mammographie ou ultrasound sein)",
            "Skin (Photo dermatologique – cancer de la peau)"
        ])

        model_files = {
            "Lung (CT scan poumon)": "best_model.hdf5",
            "Breast (Mammographie ou ultrasound sein)": "breast_cancer_model.h5",
            "Skin (Photo dermatologique – cancer de la peau)": "skin_cancer_model.h5"
        }

        classes_dict = {
            "Lung (CT scan poumon)": ["Normal (Pas de cancer)", "Adénocarcinome", "Carcinome à grandes cellules", "Carcinome épidermoïde"],
            "Breast (Mammographie ou ultrasound sein)": ["Bénin", "Malin"],
            "Skin (Photo dermatologique – cancer de la peau)": ["Bénin", "Malin"]
        }

        input_sizes = {
            "Lung (CT scan poumon)": (299, 299),
            "Breast (Mammographie ou ultrasound sein)": (224, 224),
            "Skin (Photo dermatologique – cancer de la peau)": (224, 224)
        }

        @st.cache_resource
        def load_image_model(cancer_type):
            model_path = model_files[cancer_type]

            if "Lung" in cancer_type:
                base = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
                x = base.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.5)(x)
                outputs = Dense(4, activation='softmax')(x)
                model = Model(base.input, outputs)
                for layer in base.layers: layer.trainable = False
                try:
                    model.load_weights(model_path)
                    st.success("Modèle poumon chargé.")
                except: st.warning("Poids non appliqués → modèle ImageNet.")
                return model, "xception"

            elif "Skin" in cancer_type:
                try:
                    model = tf.keras.models.load_model(model_path)
                    return model, "simple"
                except:
                    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                    x = base.output
                    x = GlobalAveragePooling2D()(x)
                    outputs = Dense(2, activation='softmax')(x)
                    model = Model(base.input, outputs)
                    try:
                        model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    except: pass
                    return model, "simple"

            else:  # Breast
                try:
                    model = tf.keras.models.load_model(model_path)
                    return model, "simple"
                except Exception as e:
                    st.error(f"Erreur breast : {e}")
                    st.stop()

        image_model, preprocess_mode = load_image_model(cancer_type_selected)
        classes = classes_dict[cancer_type_selected]
        target_size = input_sizes[cancer_type_selected]

        uploaded_file = st.file_uploader("Uploader une image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img_original = Image.open(uploaded_file).convert("RGB")
            st.image(img_original, caption="Image originale", use_column_width=True)

            processed_img = img_original
            if "Skin" in cancer_type_selected:
                w, h = img_original.size
                crop = int(min(w, h) * 0.9)
                left = (w - crop) // 2
                top = (h - crop) // 2
                processed_img = img_original.crop((left, top, left + crop, top + crop))
                st.image(processed_img, caption="Crop centré sur lésion", use_column_width=True)

            img_resized = processed_img.resize(target_size)
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            if preprocess_mode == "xception":
                from tensorflow.keras.applications.xception import preprocess_input
                img_array = preprocess_input(img_array)
            else:
                img_array /= 255.0

            if st.button("🔍 Analyser l'image", type="primary"):
                with st.spinner("Prédiction..."):
                    pred = image_model.predict(img_array)[0]
                    confidence = np.max(pred) * 100
                    idx = np.argmax(pred)
                    result = classes[idx]

                    st.success(f"**Prédiction : {result}**")
                    st.info(f"**Confiance : {confidence:.2f}%**")

                    st.markdown("### Probabilités")
                    for i, p in enumerate(pred):
                        st.progress(float(p))
                        st.caption(f"{classes[i]} : {p*100:.2f}%")

                    risk = "Faible" if any(b in result for b in ["Normal", "Bénin"]) else "Élevé" if confidence >= 80 else "Moyen" if confidence >= 50 else "Incertain"
                    st.markdown(f"### Niveau de risque : **{risk}**")

                    st.subheader("Recommandations")
                    st.markdown("- 🛑 **Consultez un médecin spécialiste immédiatement.**")
                    st.markdown("- ⚠️ Outil éducatif – pas un diagnostic.")
                    if "Malin" in result or "Adéno" in result or "Carcinome" in result:
                        st.markdown("- ❗ Risque de malignité détecté.")
                    st.markdown("**Prévention :** arrêt tabac • alimentation équilibrée • activité physique • protection solaire")