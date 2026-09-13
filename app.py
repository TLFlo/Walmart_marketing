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
def load_ml_artifacts_local():
    """Charge le dictionnaire (modèle + liste des features) depuis le fichier local churn_model.pkl."""
    model_path = "churn_model.pkl"

    if os.path.exists(model_path):
        try:
            artifacts = joblib.load(model_path)
            model = artifacts.get("model")
            features = artifacts.get("features")
            return model, features
        except Exception as e:
            st.error(
                f"Erreur lors de la lecture du fichier `churn_model.pkl` : {e}"
            )
            return None, None
    else:
        st.error(
            "❌ **Fichier `churn_model.pkl` introuvable !** "
            "Assurez-vous qu'il est bien placé à la racine du dossier du projet."
        )
        return None, None


# Exécution du chargement local
model, model_features = load_ml_artifacts_local()

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
    st.subheader("🔮 Prédiction du Churn Client en Temps Réel")
    st.markdown(
        "Remplissez les informations principales du client ou **sélectionnez un profil type** pour calculer le risque de départ."
    )

    # --- Boutons de simulation rapide (Profil A / Profil B) ---
    st.markdown("##### 🧪 Charger un profil de test rapide")
    demo_col1, demo_col2, demo_col3 = st.columns([1, 1, 2])

    preset = None
    with demo_col1:
        if st.button("👤 Profil A (Client Fidèle)"):
            preset = "Profil_A"
    with demo_col2:
        if st.button("⚠️ Profil B (Risque Churn)"):
            preset = "Profil_B"

    # Valeurs par défaut basées sur le choix
    if preset == "Profil_A":
        def_age, def_spent, def_freq, def_qty = 32, 1850.0, 12, 24
        def_tenure, def_span, def_uniq = 300, 280, 10
        def_gender, def_online = "Homme", True
    elif preset == "Profil_B":
        def_age, def_spent, def_freq, def_qty = 48, 45.0, 1, 1
        def_tenure, def_span, def_uniq = 450, 0, 1
        def_gender, def_online = "Femme", False
    else:
        def_age, def_spent, def_freq, def_qty = 35, 500.0, 5, 10
        def_tenure, def_span, def_uniq = 180, 120, 4
        def_gender, def_online = "Homme", True

    # --- Formulaire de saisie utilisateur ---
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Démographie & Ancienneté")
            age = st.number_input(
                "Âge", min_value=18, max_value=100, value=def_age
            )
            gender_str = st.selectbox(
                "Genre",
                ["Homme", "Femme"],
                index=0 if def_gender == "Homme" else 1,
            )
            is_online_shopper = st.checkbox(
                "Acheteur en ligne habituel", value=def_online
            )
            tenure_days = st.number_input(
                "Jours depuis l'inscription (Tenure Days)",
                min_value=1,
                value=def_tenure,
            )
            customer_span_days = st.number_input(
                "Durée d'activité (Span Days)",
                min_value=0,
                value=def_span,
                help="Nombre de jours entre le 1er et le dernier achat",
            )

        with col2:
            st.markdown("#### 🛒 Comportement d'Achat")
            total_spent = st.number_input(
                "Montant Total Dépensé ($)",
                min_value=0.0,
                value=float(def_spent),
            )
            frequency = st.number_input(
                "Nombre total de commandes (Fréquence)",
                min_value=1,
                value=def_freq,
            )
            total_quantity = st.number_input(
                "Quantité totale d'articles achetés",
                min_value=1,
                value=def_qty,
            )
            unique_products = st.number_input(
                "Nombre de produits uniques achetés",
                min_value=1,
                value=def_uniq,
            )

        submit_btn = st.form_submit_button(
            "🚀 Calculez le risque de Churn", use_container_width=True
        )

    # --- Calculs & Prédictions ---
    if submit_btn:
        if model is not None and model_features is not None:
            # Encoding basique
            gender_encoded = 1 if gender_str == "Homme" else 0
            online_encoded = 1 if is_online_shopper else 0

            # Calcul automatique des métriques dérivées (Feature Engineering)
            avg_basket = total_spent / frequency if frequency > 0 else 0
            purchase_rate = frequency / (tenure_days + 1)
            avg_item_price = (
                total_spent / total_quantity if total_quantity > 0 else 0
            )
            monetary_velocity = total_spent / (tenure_days + 1)
            basket_depth = (
                unique_products / (frequency + 1)
            )  # Ratio d'exploration du catalogue

            # Construction du DataFrame avec exactement les clés attendues par le modèle
            input_dict = {
                "Age": age,
                "Total_Spent": total_spent,
                "Frequency": frequency,
                "Avg_Basket": avg_basket,
                "Total_Quantity": total_quantity,
                "Tenure_Days": tenure_days,
                "Customer_Span_Days": customer_span_days,
                "Purchase_Rate": purchase_rate,
                "Avg_Item_Price": avg_item_price,
                "Unique_Products": unique_products,
                "Monetary_Velocity": monetary_velocity,
                "Basket_Depth": basket_depth,
                "Gender_Encoded": gender_encoded,
                "Is_Online_Shopper": online_encoded,
            }

            input_df = pd.DataFrame([input_dict])

            # Assurer l'ordre exact des colonnes requis par model_features
            input_df = input_df[model_features]

            # Prédiction & Probabilité
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100

            st.divider()

            # --- Affichage des Résultats ---
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if probability >= 50:
                    st.error(
                        f"⚠️ **Diagnostic : Risque Élevé de Départ ({probability:.1f}%)**"
                    )
                    st.write(
                        "Ce profil montre un décrochage net d'activité par rapport à son ancienneté."
                    )
                    st.info(
                        "💡 **Recommandation Marketing :** Déclencher une campagne de ré-engagement par e-mail avec un bon de réduction personnalisé."
                    )
                else:
                    st.success(
                        f"✅ **Diagnostic : Client Fidèle (Risque de Churn : {probability:.1f}%)**"
                    )
                    st.write(
                        "Le client présente un comportement d'achat régulier et une bonne vélocité monétaire."
                    )
                    st.info(
                        "💡 **Recommandation Marketing :** Proposer un programme VIP / Vente croisée (Cross-selling)."
                    )

            with res_col2:
                st.metric(
                    label="Score de Churn",
                    value=f"{probability:.1f} %",
                    delta="Élevé" if probability >= 50 else "Faible",
                    delta_color="inverse" if probability >= 50 else "normal",
                )
        else:
            st.error(
                "❌ Impossible d'effectuer la prédiction : Le modèle ou la liste des caractéristiques n'ont pas pu être chargés depuis le fichier `churn_model.pkl`."
            )