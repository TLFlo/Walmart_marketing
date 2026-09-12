import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONFIGURATION DE LA PAGE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Walmart Marketing & ML Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. PARAMÈTRES ET CONFIGURATION
# ==========================================
# ⚠️ Remplacez l'URL ci-dessous par votre lien "Publier sur le web" de Power BI Service
POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiZmIxZTcxZjEtYjllNi00ODY0LTkxOWItNGYyNzMzZGMwMGNlIiwidCI6ImQwMzAyZmNjLTNlODEtNDljMy04MjM1LWQzMTFhMzY4NGNmYyJ9"


@st.cache_resource
def load_ml_model():
    """Charge le modèle de Machine Learning entraîné (ex: Churn / CLV)."""
    model_path = "model_churn.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_ml_model()

# ==========================================
# 3. BARRE LATÉRALE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/1/14/Walmart_Spark.svg",
        width=50,
    )
    st.title("Navigation")
    st.info(
        "Projet d'Analyse Marketing & Machine Learning pour la fidélisation client."
    )
    st.divider()
    st.caption("Projet Académique — Modules M2 à M8")

# ==========================================
# 4. EN-TÊTE PRINCIPAL
# ==========================================
st.title("🛒 Walmart Marketing Dashboard & ML Predictions")
st.markdown(
    "Cette plateforme centralise l'analyse des campagnes marketing, la segmentation client et la prédiction du risque de départ (Churn)."
)

# Création des deux onglets principaux
tab1, tab2 = st.tabs(
    ["📊 Dashboard Power BI (M8)", "🔮 Module de Prédiction ML (M6)"]
)

# ==========================================
# ONGLET 1 : DASHBOARD POWER BI INTERACTIF
# ==========================================
with tab1:
    st.subheader("Analyse Globale des Ventes & Campagnes Marketing")
    st.write(
        "Explorez les KPIs interactifs, le suivi des ventes et l'efficacité des campagnes."
    )

    if (
        "eyJrIjoi..." in POWERBI_URL
        or "TON_CODE_POWERBI_ICI" in POWERBI_URL
    ):
        st.warning(
            "⚠️ Veillez à remplacer la variable `POWERBI_URL` dans le fichier `app.py` par votre vrai lien d'intégration Power BI Service (Fichier > Intégrer le rapport > Publier sur le web)."
        )
    else:
        # Intégration responsive via Iframe
        st.components.v1.iframe(POWERBI_URL, height=780, scrolling=True)

# ==========================================
# ONGLET 2 : FORMULAIRE DE PRÉDICTION ML
# ==========================================
with tab2:
    st.subheader("Prédiction du Churn Client en Temps Réel")
    st.markdown(
        "Remplissez les informations du client pour évaluer son risque de départ à l'aide du modèle pré-entraîné."
    )

    # Formulaire de saisie des caractéristiques (Features)
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Profil Client")
            age = st.number_input(
                "Âge du client", min_value=18, max_value=100, value=35
            )
            anciennete = st.number_input(
                "Ancienneté (Mois)", min_value=0, max_value=120, value=24
            )
            recence = st.number_input(
                "Jours depuis le dernier achat (Récence)",
                min_value=0,
                max_value=365,
                value=30,
            )

        with col2:
            st.markdown("#### 🛒 Historique d'Achat & Marketing")
            frequence = st.number_input(
                "Nombre total de commandes (Fréquence)",
                min_value=1,
                max_value=200,
                value=8,
            )
            montant_total = st.number_input(
                "Dépenses totales ($)", min_value=0.0, value=450.0
            )
            campagnes_cliquees = st.number_input(
                "Nombre de campagnes réagies",
                min_value=0,
                max_value=20,
                value=3,
            )

        submit_btn = st.form_submit_button(
            "🚀 Lancer la Prédiction", use_container_width=True
        )

    # Logique de calcul du résultat
    if submit_btn:
        if model is not None:
            # Construction du tableau de données (doit correspondre à l'ordre des features d'entraînement)
            input_features = np.array(
                [[age, anciennete, recence, frequence, montant_total, campagnes_cliquees]]
            )

            # Calcul de la prédiction
            prediction = model.predict(input_features)[0]

            # Vérification de la disponibilité des probabilités
            has_proba = hasattr(model, "predict_proba")
            probability = (
                model.predict_proba(input_features)[0][1] * 100
                if has_proba
                else None
            )

            st.divider()

            # Affichage visuel des résultats
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if prediction == 1:
                    st.error("⚠️ **Diagnostic : Risque élevé de Churn !**")
                    st.write(
                        "Ce client présente un comportement similaire aux clients ayant abandonné la plateforme."
                    )
                    st.info(
                        "💡 **Action Recommandée :** Attribuer un bon de réduction ciblé ou proposer une offre de fidélisation prioritaire."
                    )
                else:
                    st.success("✅ **Diagnostic : Client Fidèle**")
                    st.write(
                        "Le client présente un engagement satisfaisant et un faible risque de départ."
                    )
                    st.info(
                        "💡 **Action Recommandée :** Intégrer au programme ambassadeur et proposer du Cross-selling."
                    )

            with res_col2:
                if probability is not None:
                    st.metric(
                        label="Probabilité de départ",
                        value=f"{probability:.1f} %",
                        delta=f"{'- High Risk' if prediction == 1 else 'Low Risk'}",
                        delta_color="inverse" if prediction == 1 else "normal",
                    )
        else:
            st.error(
                "❌ **Fichier modèle introuvable.** Assurez-vous que le fichier `model_churn.pkl` est présent à la racine de votre dossier."
            )