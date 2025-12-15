import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Outil d'Évaluation des Risques de Cancer – Projet Universitaire",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DISCLAIMER GLOBAL
# ============================================================
st.markdown("""
<style>
.big-warning {
    font-size: 18px !important;
    color: #d32f2f;
    background-color: #ffebee;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #d32f2f;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

disclaimer = """
<div class="big-warning">
🛑 <strong>AVERTISSEMENT IMPORTANT – OUTIL ÉDUCATIF ET ACADÉMIQUE UNIQUEMENT</strong><br><br>
Cet outil a été développé dans le cadre d'un projet universitaire pour démontrer l'application de l'intelligence artificielle en santé publique.<br>
Il <strong>NE CONSTITUE PAS</strong> un diagnostic médical, un conseil médical ou un substitut à une consultation professionnelle.<br>
Les prédictions sont basées sur des modèles d'apprentissage automatique entraînés sur des données publiques et ne remplacent en aucun cas l'avis d'un médecin qualifié.<br>
<strong>Toujours consulter un professionnel de santé pour toute question relative à votre santé.</strong>
</div>
"""
st.markdown(disclaimer, unsafe_allow_html=True)

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
# CHARGEMENT DES MODÈLES
# ============================================================


@st.cache_resource
def load_risk_model():
    try:
        return joblib.load("model_reco.pkl")
    except Exception as e:
        st.error(f"Erreur chargement modèle risque : {e}")
        st.stop()


@st.cache_resource
def load_cancer_model():
    try:
        return joblib.load("model_cancer_type.pkl")
    except Exception as e:
        st.error(f"Erreur chargement modèle type cancer : {e}")
        st.stop()


@st.cache_resource
def load_encoder():
    try:
        return joblib.load("cancer_type_encoder.pkl")
    except Exception as e:
        st.error(f"Erreur chargement encodeur : {e}")
        st.stop()


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
    try:
        value = float(value)
    except ValueError:
        return False, "⚠️ Veuillez entrer une valeur numérique valide."
    rule = VALIDATION_RULES.get(feature)
    if isinstance(rule, tuple):
        if not rule[0] <= value <= rule[1]:
            return False, f"⚠️ Valeur doit être entre {rule[0]} et {rule[1]}."
    elif isinstance(rule, list) and value not in rule:
        return False, f"⚠️ Valeurs autorisées : {', '.join(map(str, rule))}."
    return True, value


def generate_recommendations(row, predicted_risk, cancer_type):
    recos = []
    def has(col): return col in row.index

    if has('Smoking') and row['Smoking'] >= 7:
        recos.append(
            "🛑 Arrêt du tabac fortement recommandé – consultez un tabacologue.")
    if has('Alcohol_Use') and row['Alcohol_Use'] >= 7:
        recos.append(
            "Limitez l'alcool (<1 verre/jour pour femmes, <2 pour hommes).")
    if has('BMI') and row['BMI'] >= 30:
        recos.append(
            "Objectif : IMC < 25 – adoptez une alimentation équilibrée et activité physique.")
    if has('Fruit_Veg_Intake') and row['Fruit_Veg_Intake'] <= 3:
        recos.append("Consommez ≥5 portions de fruits/légumes par jour.")
    if has('Physical_Activity') and row['Physical_Activity'] <= 3:
        recos.append("≥150 minutes d'activité modérée par semaine.")
    if has('Family_History') and row['Family_History'] == 1:
        recos.append(
            "Dépistage précoce recommandé en raison des antécédents familiaux.")

    risk_lower = str(predicted_risk).lower()
    if "high" in risk_lower or "élevé" in risk_lower:
        recos.append(
            "🛑 **Risque élevé** – Consultation médicale urgente recommandée.")
    elif "medium" in risk_lower or "moderate" in risk_lower or "moyen" in risk_lower:
        recos.append(
            "⚠️ **Risque moyen** – Appliquez les changements de mode de vie.")
    else:
        recos.append("✅ **Risque faible** – Maintenez un mode de vie sain.")

    cancer_lower = cancer_type.lower()
    if "breast" in cancer_lower or "sein" in cancer_lower:
        recos.append(
            "Mammographie régulière recommandée à partir de 40-50 ans.")
    elif "lung" in cancer_lower or "poumon" in cancer_lower:
        recos.append(
            "Scanner thoracique basse dose si antécédents de tabagisme important.")
    elif "skin" in cancer_lower or "peau" in cancer_lower:
        recos.append("Protection solaire et examen dermatologique annuel.")

    recos.append(
        "**Prévention générale** : Alimentation saine, activité physique, éviter tabac/alcool excessif.")
    return list(dict.fromkeys(recos))


def recommend_for_patient(patient_features):
    df_cancer = pd.DataFrame([patient_features])[CANCER_TYPE_FEATURES]
    pred_encoded = cancer_model.predict(df_cancer)[0]
    cancer_type = cancer_type_encoder.inverse_transform([pred_encoded])[0]
    patient_features['Cancer_Type'] = cancer_type
    df_risk = pd.DataFrame([patient_features])[RECO_FEATURES]
    risk_pred = risk_model.predict(df_risk)[0]
    recos = generate_recommendations(df_risk.iloc[0], risk_pred, cancer_type)
    return cancer_type, risk_pred, recos


# ============================================================
# FONCTIONS PDF – CORRIGÉES ET PLUS PROFESSIONNELLES
# ============================================================
LOGO_PATH = "./assets/logo.png"  # Assurez-vous que ce fichier existe


def generate_questionnaire_pdf(cancer_type, risk_pred, recos):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm,
                            leftMargin=2*cm, topMargin=1.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor("#2E4057")
    )

    subtitle_style = ParagraphStyle(
        name='Subtitle',
        fontSize=12,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=colors.grey
    )

    heading_style = ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1A5276")
    )

    story = []

    # Logo – utilisation directe du chemin (pas ImageReader ici)
    try:
        logo = RLImage(LOGO_PATH, width=4*cm, height=4*cm)
        logo.hAlign = 'CENTER'
        story.append(logo)
        story.append(Spacer(1, 0.5*cm))
    except Exception as e:
        st.warning(f"Impossible d'ajouter le logo au PDF : {e}")

    story.append(
        Paragraph("Rapport d'Évaluation des Risques de Cancer", title_style))
    story.append(Paragraph(
        "Outil d'Intelligence Artificielle à visée éducative", subtitle_style))
    story.append(Paragraph(
        f"Date du rapport : {datetime.now().strftime('%d %B %Y à %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Résultats Principaux", heading_style))
    data = [
        ["Type de cancer le plus probable :", cancer_type],
        ["Niveau de risque estimé :", risk_pred],
    ]
    result_table = Table(data, colWidths=[7*cm, 9*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EBF5FB")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Recommandations Personnalisées", heading_style))
    for r in recos:
        story.append(Paragraph(f"• {r}", styles['Normal']))
        story.append(Spacer(1, 0.4*cm))

    story.append(PageBreak())

    story.append(Paragraph("Avertissement Important", heading_style))
    story.append(Paragraph(
        "Ce rapport est généré par un modèle d'intelligence artificielle à des fins éducatives et de sensibilisation uniquement.<br/>"
        "Il ne constitue en aucun cas un diagnostic médical ni un conseil thérapeutique.<br/>"
        "Toute décision concernant votre santé doit être prise en consultation avec un professionnel de santé qualifié.",
        styles['Normal']
    ))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Projet Universitaire 2025", styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_image_pdf(cancer_type_selected, result, confidence, risk, pred_probs, classes):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm,
                            leftMargin=2*cm, topMargin=1.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor("#2E4057")
    )

    subtitle_style = ParagraphStyle(
        name='Subtitle',
        fontSize=12,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=colors.grey
    )

    heading_style = ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1A5276")
    )

    story = []

    try:
        logo = RLImage(LOGO_PATH, width=4*cm, height=4*cm)
        logo.hAlign = 'CENTER'
        story.append(logo)
        story.append(Spacer(1, 0.5*cm))
    except Exception as e:
        st.warning(f"Impossible d'ajouter le logo au PDF : {e}")

    story.append(
        Paragraph("Rapport d'Analyse d'Image Médicale par IA", title_style))
    story.append(Paragraph(
        "Outil d'Intelligence Artificielle à visée éducative", subtitle_style))
    story.append(Paragraph(
        f"Date du rapport : {datetime.now().strftime('%d %B %Y à %H:%M')}", styles['Normal']))
    story.append(
        Paragraph(f"Type d'image analysée : {cancer_type_selected}", styles['Normal']))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Résultats de l'Analyse", heading_style))
    data = [
        ["Prédiction principale :", result],
        ["Confiance du modèle :", f"{confidence:.2f}%"],
        ["Niveau de risque estimé :", risk],
    ]
    result_table = Table(data, colWidths=[7*cm, 9*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EBF5FB")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 1*cm))

    story.append(
        Paragraph("Probabilités Détaillées par Classe", heading_style))
    prob_data = [["Classe", "Probabilité (%)"]]
    for i, p in enumerate(pred_probs):
        prob_data.append([classes[i], f"{p*100:.2f}"])
    prob_table = Table(prob_data, colWidths=[9*cm, 7*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A5276")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(prob_table)

    story.append(PageBreak())

    story.append(Paragraph("Avertissement Important", heading_style))
    story.append(Paragraph(
        "Ce rapport est généré par un modèle d'intelligence artificielle à des fins éducatives et de démonstration uniquement.<br/>"
        "Il ne remplace en aucun cas un diagnostic médical réalisé par un radiologue ou un médecin spécialiste.<br/>"
        "Toute suspicion de pathologie doit faire l'objet d'un examen médical approfondi.",
        styles['Normal']
    ))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "Projet Universitaire 2025", styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
try:
    st.sidebar.image(LOGO_PATH, width=100)
except:
    st.sidebar.image(
        "https://png.pngtree.com/png-clipart/20250524/original/pngtree-3d-pink-ribbon-png-clipart-breast-cancer-awareness-png-image_21063411.png", width=80)

st.sidebar.title("OncoRisk AI")
page = st.sidebar.radio("Pages", [
    "Accueil & Introduction",
    "À Propos du Projet",
    "Évaluation par Questionnaire",
    "Analyse d'Images Médicales",
    "Références & Sources"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Projet Universitaire** – Intelligence Artificielle Appliquée à la Prévention du Cancer")
st.sidebar.caption("""
**Supervision :** Mme, Oumaima Guesmi  
**Équipe de réalisation :**  
- Ahmed Trabelsi  
- Samar Omrani  
- Ahmed Fekih  
- Malek Hammami  
- Ikbel Hamdi  
- Maram Rachdi  

**Année universitaire : 2025**
""")

# ============================================================
# PAGES
# ============================================================
if page == "Accueil & Introduction":
    st.title("🩺 Outil d'Évaluation des Risques de Cancer")
    st.markdown("""
    Cet outil académique propose deux approches complémentaires pour sensibiliser aux risques de cancer :
    
    - **Questionnaire interactif** : Évaluation basée sur les facteurs de risque modifiables et non modifiables.
    - **Analyse d'images médicales** : Démonstration de modèles de Transfer Learning pour la détection précoce (poumon, sein, peau).
    
    Les modèles sont entraînés sur des datasets publics (ex. Kaggle, IQ-OTH/NCCD pour poumon).
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://static.vecteezy.com/system/resources/previews/029/338/717/non_2x/medical-illustration-concept-of-lung-cancer-prevention-diet-food-stop-smoking-avoid-secondhand-smoke-avoid-carcinogens-tested-for-radon-at-home-isolated-on-white-background-flat-style-vector.jpg", caption="Prévention cancer du poumon")
    with col2:
        st.image("https://static.vecteezy.com/system/resources/previews/027/607/078/non_2x/medical-illustration-breast-cancer-prevention-do-not-smoke-limit-alcohol-breastfeed-get-enough-rest-control-your-weight-eat-a-healthy-diet-exercise-regularly-illustrations-flat-style-vector.jpg", caption="Prévention cancer du sein")
    with col3:
        st.image("https://nutritionsource.hsph.harvard.edu/wp-content/uploads/2021/03/aicr-cancer-prevention-recommendations-scaled-1.png",
                 caption="Recommandations AICR pour la prévention")

elif page == "À Propos du Projet":
    st.title("À Propos du Projet Universitaire")
    st.markdown("""
    ### Objectifs
    - Démontrer l'utilisation de l'IA pour la sensibilisation et la prévention du cancer.
    - Combiner modèles classiques (Random Forest, AdaBoost) et Deep Learning (Transfer Learning).
    - Promouvoir les bonnes pratiques en santé publique.
    
    ### Méthodologie
    - **Questionnaire** : Modèles ML entraînés sur données synthétiques/simulées basées sur facteurs validés (ACS, WHO).
    - **Images** : Transfer Learning avec Xception (poumon) et EfficientNetB0 (sein/peau).
    
    ### Limites
    - Outil non validé cliniquement.
    - Précision dépendante des données d'entraînement.
    """)

elif page == "Évaluation par Questionnaire":
    st.title("Questionnaire Interactif – Facteurs de Risque")

    if "current_conv" not in st.session_state:
        st.session_state.current_conv = {
            "messages": [{"role": "assistant", "content": "Bienvenue ! Je vais vous poser des questions pour évaluer vos facteurs de risque."}],
            "responses": {},
            "question_index": 0,
            "completed": False
        }

    conv = st.session_state.current_conv
    questions = {
        'Age': "Quel est votre âge ? (0-120)",
        'Gender': "Sexe ? (0 = Femme, 1 = Homme)",
        'Smoking': "Niveau de tabagisme ? (0=Non-fumeur, 10=Fumeur intensif)",
        'Alcohol_Use': "Consommation d'alcool ? (0=Aucune, 10=Intensive)",
        'Obesity': "Niveau d'obésité perçue ? (0=Aucun, 10=Sévère)",
        'Family_History': "Antécédents familiaux de cancer ? (0=Non, 1=Oui)",
        'Diet_Red_Meat': "Consommation de viande rouge ? (0=Faible, 10=Élevée)",
        'Diet_Salted_Processed': "Consommation d'aliments salés/transformés ? (0=Faible, 10=Élevée)",
        'Fruit_Veg_Intake': "Consommation de fruits/légumes ? (0=Faible, 10=Élevée)",
        'Physical_Activity': "Niveau d'activité physique ? (0=Inactif, 10=Très actif)",
        'Air_Pollution': "Exposition à la pollution atmosphérique ? (0=Faible, 10=Élevée)",
        'Occupational_Hazards': "Exposition à des hazards professionnels ? (0=Faible, 10=Élevée)",
        'BRCA_Mutation': "Mutation BRCA connue ? (0=Non, 1=Oui)",
        'H_Pylori_Infection': "Infection à H. pylori ? (0=Non, 1=Oui)",
        'Calcium_Intake': "Apport en calcium ? (0=Faible, 10=Élevé)",
        'Overall_Risk_Score': "Score de risque global connu ? (0-1, sinon entrez 0)",
        'BMI': "Quel est votre IMC ? (10-60)",
        'Physical_Activity_Level': "Niveau global d'activité physique ? (0=Faible, 10=Élevé)"
    }
    keys = list(questions.keys())

    chat_container = st.container()
    with chat_container:
        for msg in conv["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if conv.get("completed", False):
        st.success("Évaluation terminée !")
        cancer_type, risk_pred, recos = conv["results"]

        st.info(f"**Type de cancer prédit :** {cancer_type}")
        st.warning(f"**Niveau de risque estimé :** {risk_pred}")
        st.subheader("Recommandations personnalisées")
        for r in recos:
            st.markdown(f"• {r}")

        pdf_data = generate_questionnaire_pdf(cancer_type, risk_pred, recos)

        st.download_button(
            label="📥 Télécharger le rapport médical (PDF)",
            data=pdf_data,
            file_name=f"rapport_risque_cancer_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

        if st.button("Nouvelle évaluation"):
            st.session_state.current_conv = {
                "messages": [{"role": "assistant", "content": "Nouvelle évaluation démarrée !"}],
                "responses": {}, "question_index": 0, "completed": False
            }
            st.rerun()

    else:
        progress = conv["question_index"] / len(keys)
        st.progress(progress)
        st.caption(f"Question {conv['question_index'] + 1} sur {len(keys)}")

        if conv["question_index"] < len(keys):
            key = keys[conv["question_index"]]
            q = questions[key]
            if not conv["messages"] or conv["messages"][-1]["content"] != q:
                with chat_container.chat_message("assistant"):
                    st.markdown(q)
                conv["messages"].append({"role": "assistant", "content": q})
                st.rerun()

            if user_input := st.chat_input("Entrez votre réponse ici..."):
                valid, msg_or_value = validate_input(key, user_input)
                if valid:
                    conv["responses"][key] = msg_or_value
                    with chat_container.chat_message("user"):
                        st.markdown(user_input)
                    conv["messages"].append(
                        {"role": "user", "content": user_input})
                    conv["question_index"] += 1
                    st.rerun()
                else:
                    with chat_container.chat_message("assistant"):
                        st.error(msg_or_value)
                    conv["messages"].append(
                        {"role": "assistant", "content": msg_or_value})
                    st.rerun()
        else:
            cancer_type, risk_pred, recos = recommend_for_patient(
                conv["responses"])
            conv["results"] = (cancer_type, risk_pred, recos)
            conv["completed"] = True
            with chat_container.chat_message("assistant"):
                st.markdown(
                    f"**Type de cancer prédit :** {cancer_type}\n**Niveau de risque :** {risk_pred}\n**Recommandations :**")
                for r in recos:
                    st.markdown(f"- {r}")
            conv["messages"].append({
                "role": "assistant",
                "content": f"Résultats : {cancer_type}, {risk_pred}\n" + "\n".join(f"- {r}" for r in recos)
            })
            st.rerun()

elif page == "Analyse d'Images Médicales":
    st.title("Analyse d'Images Médicales (Démonstration IA)")
    st.markdown("### Exemples illustratifs")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://www.researchgate.net/publication/333538102/figure/fig1/AS:765048309964805@1559413139628/CT-scan-image-of-lung-normal-and-lung-diseases-caused-by-smoking-a-lung-normal-b.jpg",
                 caption="Exemple CT poumon normal vs pathologique")
    with col2:
        st.image("https://news.mit.edu/sites/default/files/images/201905/BreastCancerAI.png",
                 caption="Exemple mammographie avec détection IA")

    st.markdown("### 🔬 Analyse d'une image médicale")
    st.warning(
        "⚠️ **Outil éducatif uniquement** • Ne remplace PAS un diagnostic médical • Consultez un spécialiste.")

    cancer_type_selected = st.selectbox("Type d'image :", [
        "Poumon (CT scan)",
        "Sein (Mammographie ou échographie)",
        "Peau (Photo dermatologique)"
    ])

    model_files = {
        "Poumon (CT scan)": "best_model.hdf5",
        "Sein (Mammographie ou échographie)": "breast_cancer_model.h5",
        "Peau (Photo dermatologique)": "skin_cancer_model.h5"
    }

    classes_dict = {
        "Poumon (CT scan)": ["Normal (Pas de cancer)", "Adénocarcinome", "Carcinome à grandes cellules", "Carcinome épidermoïde"],
        "Sein (Mammographie ou échographie)": ["Bénin", "Malin"],
        "Peau (Photo dermatologique)": ["Bénin", "Malin"]
    }

    input_sizes = {
        "Poumon (CT scan)": (299, 299),
        "Sein (Mammographie ou échographie)": (224, 224),
        "Peau (Photo dermatologique)": (224, 224)
    }

    preprocess_modes = {
        "Poumon (CT scan)": "xception",
        "Sein (Mammographie ou échographie)": "efficientnet",
        "Peau (Photo dermatologique)": "efficientnet"
    }

    @st.cache_resource
    def load_image_model(cancer_type):
        model_path = model_files[cancer_type]
        input_shape = (*input_sizes[cancer_type], 3)

        if "Poumon" in cancer_type:
            base = Xception(weights='imagenet',
                            include_top=False, input_shape=input_shape)
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(
                len(classes_dict[cancer_type]), activation='softmax')(x)
            model = Model(inputs=base.input, outputs=outputs)
            for layer in base.layers:
                layer.trainable = False
            try:
                model.load_weights(model_path)
                st.success("Modèle pour poumon chargé avec succès.")
            except Exception as e:
                st.warning(
                    f"Impossible de charger les poids : {e}. Utilisation du modèle de base.")
            return model, preprocess_modes[cancer_type]
        else:
            base = EfficientNetB0(weights='imagenet',
                                  include_top=False, input_shape=input_shape)
            x = base.output
            x = GlobalAveragePooling2D()(x)
            outputs = Dense(
                len(classes_dict[cancer_type]), activation='softmax')(x)
            model = Model(inputs=base.input, outputs=outputs)
            try:
                model.load_weights(model_path, by_name=True,
                                   skip_mismatch=True)
                st.success(f"Modèle chargé avec succès.")
            except Exception as e:
                st.warning(
                    f"Impossible de charger les poids : {e}. Utilisation du modèle de base.")
            return model, preprocess_modes[cancer_type]

    image_model, preprocess_mode = load_image_model(cancer_type_selected)
    classes = classes_dict[cancer_type_selected]
    target_size = input_sizes[cancer_type_selected]

    uploaded_file = st.file_uploader(
        "Uploader une image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_original = PILImage.open(uploaded_file).convert("RGB")
        st.image(img_original, caption="Image originale",
                 use_column_width=True)

        processed_img = img_original
        if "Peau" in cancer_type_selected:
            w, h = img_original.size
            crop = int(min(w, h) * 0.9)
            left = (w - crop) // 2
            top = (h - crop) // 2
            processed_img = img_original.crop(
                (left, top, left + crop, top + crop))
            st.image(processed_img, caption="Crop centré sur lésion",
                     use_column_width=True)

        img_resized = processed_img.resize(target_size)
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        if preprocess_mode == "xception":
            img_array = xception_preprocess(img_array)
        elif preprocess_mode == "efficientnet":
            img_array = efficientnet_preprocess(img_array)
        else:
            img_array /= 255.0

        if st.button("🔍 Analyser l'image", type="primary"):
            with st.spinner("Prédiction en cours..."):
                pred = image_model.predict(img_array)[0]
                confidence = np.max(pred) * 100
                idx = np.argmax(pred)
                result = classes[idx]

                st.success(f"**Prédiction :** {result}")
                st.info(f"**Confiance :** {confidence:.2f}%")

                st.markdown("### Probabilités par classe")
                for i, p in enumerate(pred):
                    st.progress(float(p))
                    st.caption(f"{classes[i]} : {p*100:.2f}%")

                risk = "Faible" if any(b in result for b in [
                                       "Normal", "Bénin"]) else "Élevé" if confidence >= 80 else "Moyen" if confidence >= 50 else "Incertain"
                st.markdown(f"### Niveau de risque estimé : **{risk}**")

                st.subheader("Recommandations")
                st.markdown("- 🛑 **Consultez immédiatement un spécialiste.**")
                st.markdown("- ⚠️ Outil éducatif – pas un diagnostic.")
                if "Malin" in result or "Adéno" in result or "Carcinome" in result:
                    st.markdown("- ❗ Signes potentiels de malignité détectés.")
                st.markdown(
                    "**Prévention générale :** arrêt tabac • alimentation équilibrée • activité physique • protection solaire")

                pdf_data = generate_image_pdf(
                    cancer_type_selected, result, confidence, risk, pred, classes)

                st.download_button(
                    label="📥 Télécharger le rapport d'analyse (PDF)",
                    data=pdf_data,
                    file_name=f"analyse_image_{cancer_type_selected.replace(' ', '_').replace('(', '').replace(')', '')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

elif page == "Références & Sources":
    st.title("Références Scientifiques")
    st.markdown("""
    - American Cancer Society – Guidelines for Cancer Prevention
    - World Health Organization – Cancer Prevention Factsheets
    - Harvard T.H. Chan School of Public Health – Cancer Risk Index
    - Datasets : IQ-OTH/NCCD (Lung), CBIS-DDSM (Breast), ISIC (Skin)
    - Modèles : Scikit-learn, TensorFlow/Keras
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
**Projet Universitaire 2025** | Développé avec ❤️ utilisant Streamlit, scikit-learn et TensorFlow  
Ce travail est purement académique et vise à promouvoir la recherche en IA appliquée à la santé.
""")
