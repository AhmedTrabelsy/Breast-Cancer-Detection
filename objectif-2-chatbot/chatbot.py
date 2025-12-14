import streamlit as st
import joblib
import pandas as pd

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="Évaluation Risque Cancer", layout="centered")

# ============================================================
# LISTES DE FEATURES (inchangées)
# ============================================================
CANCER_TYPE_FEATURES = [
    'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity',
    'Family_History', 'Diet_Red_Meat', 'Diet_Salted_Processed',
    'Fruit_Veg_Intake', 'Physical_Activity', 'Air_Pollution',
    'Occupational_Hazards', 'BRCA_Mutation', 'H_Pylori_Infection',
    'Calcium_Intake', 'Overall_Risk_Score', 'BMI',
    'Physical_Activity_Level'
]

RECO_FEATURES = [
    'Cancer_Type', 'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity',
    'Family_History', 'Diet_Red_Meat', 'Diet_Salted_Processed',
    'Fruit_Veg_Intake', 'Physical_Activity', 'Air_Pollution',
    'Occupational_Hazards', 'BRCA_Mutation', 'H_Pylori_Infection',
    'Calcium_Intake', 'Overall_Risk_Score', 'BMI',
    'Physical_Activity_Level'
]

# ============================================================
# CHARGEMENT DES MODÈLES (inchangé)
# ============================================================
@st.cache_resource
def load_risk_model():
    return joblib.load("model_reco.pkl")

@st.cache_resource
def load_cancer_model():
    return joblib.load("model_cancer_type.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("cancer_type_encoder.pkl")

risk_model = load_risk_model()
cancer_model = load_cancer_model()
cancer_type_encoder = load_encoder()

# ============================================================
# RÈGLES DE VALIDATION & FONCTIONS (inchangées)
# ============================================================
VALIDATION_RULES = {
    'Age': (0, 120),
    'Smoking': (0, 10),
    'Alcohol_Use': (0, 10),
    'Obesity': (0, 10),
    'Diet_Red_Meat': (0, 10),
    'Diet_Salted_Processed': (0, 10),
    'Fruit_Veg_Intake': (0, 10),
    'Physical_Activity': (0, 10),
    'Air_Pollution': (0, 10),
    'Occupational_Hazards': (0, 10),
    'Calcium_Intake': (0, 10),
    'Physical_Activity_Level': (0, 10),
    'BMI': (10, 60),
    'Overall_Risk_Score': (0, 1),
    'Gender': [0, 1],
    'Family_History': [0, 1],
    'BRCA_Mutation': [0, 1],
    'H_Pylori_Infection': [0, 1]
}

def validate_input(feature, value):
    try:
        value = float(value)
    except ValueError:
        return False, "⚠️ Veuillez entrer une valeur numérique valide."
    
    rule = VALIDATION_RULES.get(feature)
    if isinstance(rule, tuple):
        min_v, max_v = rule
        if not min_v <= value <= max_v:
            return False, f"⚠️ Valeur hors plage autorisée : [{min_v} – {max_v}]."
    if isinstance(rule, list):
        if value not in rule:
            return False, f"⚠️ Valeur invalide. Valeurs possibles : {rule}"
    return True, value

def generate_recommendations(row, predicted_risk):
    # (inchangée – tu peux la garder telle quelle)
    recos = []
    def has(col):
        return col in row.index

    if has('Smoking') and row['Smoking'] >= 7:
        recos.append("Réduire progressivement puis arrêter totalement le tabac (accompagnement tabacologue, substituts nicotiniques).")
    if has('Alcohol_Use') and row['Alcohol_Use'] >= 7:
        recos.append("Limiter fortement la consommation d’alcool (max 1 verre/jour, et pas tous les jours).")
    if has('Obesity') and row['Obesity'] >= 7:
        recos.append("Mettre en place un programme de perte de poids : alimentation équilibrée, réduction des portions, suivi diététique.")
    if has('BMI') and row['BMI'] >= 30:
        recos.append("Surveiller l’IMC avec un professionnel de santé, viser un IMC < 25 si possible.")
    if has('Fruit_Veg_Intake') and row['Fruit_Veg_Intake'] <= 3:
        recos.append("Augmenter la consommation de fruits et légumes (au moins 5 portions/jour).")
    if has('Diet_Red_Meat') and row['Diet_Red_Meat'] >= 7:
        recos.append("Réduire la consommation de viandes rouges et privilégier les protéines végétales ou les viandes blanches.")
    if has('Diet_Salted_Processed') and row['Diet_Salted_Processed'] >= 7:
        recos.append("Limiter les aliments très salés et transformés (charcuterie, plats préparés).")
    if has('Physical_Activity') and row['Physical_Activity'] <= 3:
        recos.append("Augmenter l’activité physique : au moins 150 minutes/semaine d’activité modérée.")
    if has('Physical_Activity_Level') and row['Physical_Activity_Level'] <= 3:
        recos.append("Réduire la sédentarité : marcher plus, utiliser les escaliers, etc.")
    if has('Air_Pollution') and row['Air_Pollution'] >= 7:
        recos.append("Limiter l’exposition à la pollution : éviter les axes très pollués.")
    if has('Occupational_Hazards') and row['Occupational_Hazards'] >= 7:
        recos.append("Renforcer les protections au travail (EPI, ventilation).")
    if has('Family_History') and row['Family_History'] == 1:
        recos.append("Consulter un spécialiste pour un dépistage plus précoce (antécédents familiaux).")
    if has('BRCA_Mutation') and row['BRCA_Mutation'] == 1:
        recos.append("Suivi renforcé recommandé en cas de mutation génétique à risque.")
    if has('H_Pylori_Infection') and row['H_Pylori_Infection'] == 1:
        recos.append("Assurer la prise en charge de l’infection à H. pylori.")

    risk_str = str(predicted_risk).lower()
    if "high" in risk_str:
        recos.append("🛑 **Niveau de risque ÉLEVÉ** : consulter rapidement un médecin pour un bilan complet.")
    elif "medium" in risk_str or "moderate" in risk_str:
        recos.append("⚠️ **Niveau de risque MOYEN** : appliquer ces changements pour réduire le risque.")
    else:
        recos.append("✅ **Niveau de risque FAIBLE** : maintenir ce mode de vie sain.")

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
page = st.sidebar.radio("Choisir une page", [
    "Informations Générales",
    "Détails du Modèle",
    "Chatbot Prédiction"
])

# ============================================================
# PAGES 1 & 2 (inchangées)
# ============================================================
if page == "Informations Générales":
    st.title("Évaluation des risques liés au cancer")
    st.markdown("""
    Cette application utilise l'intelligence artificielle pour estimer :
    - Le **type de cancer** le plus probable
    - Le **niveau de risque** global
    - Des **recommandations personnalisées** de prévention
    """)

elif page == "Détails du Modèle":
    st.title("Modèles ML utilisés")
    st.markdown("""
    - **Prédiction du type de cancer** : Random Forest Classifier  
    - **Prédiction du niveau de risque** : AdaBoost Classifier  
    - Encodage des types de cancer via LabelEncoder  
    - Recommandations basées sur règles expertes + facteurs modifiables
    """)

# ============================================================
# PAGE 3 – CHATBOT AMÉLIORÉ
# ============================================================
else:
    st.title("🩺 Chatbot – Évaluation personnalisée du risque de cancer")

    # Initialisation de l'historique global des conversations
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_conv" not in st.session_state:
        # Créer une nouvelle conversation
        st.session_state.current_conv = {
            "messages": [],
            "responses": {},
            "question_index": 0,
            "completed": False
        }
        st.session_state.conversations.append(st.session_state.current_conv)

    conv = st.session_state.current_conv
    questions = {
        'Age': "Quel est votre âge (en années) ?",
        'Gender': "Quel est votre sexe ? (0 = Femme, 1 = Homme)",
        'Smoking': "Sur une échelle de 0 à 10, quel est votre niveau de tabagisme ? (0 = aucun, 10 = très élevé)",
        'Alcohol_Use': "Sur une échelle de 0 à 10, quel est votre niveau de consommation d’alcool ?",
        'Obesity': "Sur une échelle de 0 à 10, estimez votre niveau d’obésité.",
        'Family_History': "Avez-vous des antécédents familiaux de cancer ? (0 = Non, 1 = Oui)",
        'Diet_Red_Meat': "Sur une échelle de 0 à 10, quelle est votre consommation de viande rouge ?",
        'Diet_Salted_Processed': "Sur une échelle de 0 à 10, consommation d’aliments salés/transformés ?",
        'Fruit_Veg_Intake': "Sur une échelle de 0 à 10, consommation de fruits et légumes ?",
        'Physical_Activity': "Sur une échelle de 0 à 10, quel est votre niveau d’activité physique ?",
        'Air_Pollution': "Sur une échelle de 0 à 10, exposition à la pollution de l’air ?",
        'Occupational_Hazards': "Sur une échelle de 0 à 10, exposition à des risques professionnels ?",
        'BRCA_Mutation': "Mutation BRCA connue ? (0 = Non, 1 = Oui)",
        'H_Pylori_Infection': "Infection à H. pylori connue ? (0 = Non, 1 = Oui)",
        'Calcium_Intake': "Sur une échelle de 0 à 10, quel est votre apport en calcium ?",
        'Overall_Risk_Score': "Score de risque global connu (0 à 1) ? Sinon laissez 0.",
        'BMI': "Quel est votre IMC (BMI) ?",
        'Physical_Activity_Level': "Sur une échelle de 0 à 10, niveau global d’activité physique ?"
    }
    keys = list(questions.keys())

    # Affichage stylé du chat
    for msg in conv["messages"]:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

    # Si la conversation est terminée
    if conv.get("completed", False):
        st.success("✅ Évaluation terminée ! Voici les résultats :")
        cancer_type, risk_pred, recos = conv["results"]
        st.info(f"**Type de cancer prédit :** {cancer_type}")
        st.warning(f"**Niveau de risque :** {risk_pred}")

        st.subheader("Recommandations personnalisées")
        for r in recos:
            st.markdown(f"• {r}")

        if st.button("🔄 Nouvelle évaluation"):
            # Créer une nouvelle conversation
            new_conv = {
                "messages": [{"role": "assistant", "content": "Bonjour ! Commençons une nouvelle évaluation."}],
                "responses": {},
                "question_index": 0,
                "completed": False
            }
            st.session_state.current_conv = new_conv
            st.session_state.conversations.append(new_conv)
            st.rerun()

    else:
        # Pose la question courante
        if conv["question_index"] < len(keys):
            current_key = keys[conv["question_index"]]
            question_text = questions[current_key]

            # Afficher la question seulement si pas déjà affichée
            if not conv["messages"] or conv["messages"][-1]["role"] != "assistant" or conv["messages"][-1]["content"] != question_text:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(question_text)
                conv["messages"].append({"role": "assistant", "content": question_text})

            # Saisie utilisateur
            user_input = st.chat_input("Votre réponse ici...")

            if user_input:
                valid, feedback = validate_input(current_key, user_input)
                if valid:
                    conv["responses"][current_key] = feedback
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(user_input)
                    conv["messages"].append({"role": "user", "content": user_input})

                    conv["question_index"] += 1
                    st.rerun()
                else:
                    # Message d'erreur stylé
                    with st.chat_message("assistant", avatar="🤖"):
                        st.error(feedback)
                    conv["messages"].append({"role": "assistant", "content": feedback})
                    # On ne passe pas à la question suivante tant que ce n'est pas valide

        else:
            # Toutes les questions posées → prédiction
            cancer_type, risk_pred, recos = recommend_for_patient(conv["responses"])
            conv["results"] = (cancer_type, risk_pred, recos)
            conv["completed"] = True

            # Afficher les résultats dans le chat
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"🧬 **Type de cancer prédit :** {cancer_type}")
                st.markdown(f"⚠️ **Niveau de risque estimé :** {risk_pred}")
                st.markdown("**Recommandations personnalisées :**")
                for r in recos:
                    st.markdown(f"• {r}")
            conv["messages"].append({"role": "assistant", "content": f"Résultats :\n- Type : {cancer_type}\n- Risque : {risk_pred}\nRecommandations :\n" + "\n".join([f"• {r}" for r in recos])})

            st.rerun()