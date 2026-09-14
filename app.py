import base64
import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Marketing & ML Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. ENCODAGE DE L'IMAGE LOCALE EN BASE64
# ==========================================
LOGO_PATH = "marketing.jpeg"


def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""


logo_b64 = get_base64_of_bin_file(LOGO_PATH)
logo_src = (
    f"data:image/jpeg;base64,{logo_b64}"
    if logo_b64
    else "https://upload.wikimedia.org/wikipedia/commons/1/14/Walmart_Spark.svg"
)

# ==========================================
# 3. CSS PERSONNALISÉ (NAVBAR ALIGNÉE SUR 1 LIGNE)
# ==========================================
# ============================================================
# NAVIGATION
# ============================================================

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "dashboard"
# ============================================================
# NAVBAR
# ============================================================


col_logo, col_title, col_dashboard, col_prediction = st.columns(
    [1, 6.5, 1.5, 1.5],
    vertical_alignment="center"
)

# Logo
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=70)
    else:
        st.write("🛒")

# Titre
with col_title:
    st.markdown(
        "### Analyse & Optimisation Marketing"
    )

# Menu Dashboard
with col_dashboard:
    if st.button("Dashboard", use_container_width=True):
        st.session_state["active_tab"] = "dashboard"

# Menu Prediction
with col_prediction:
    if st.button("Prediction", use_container_width=True):
        st.session_state["active_tab"] = "prediction"

st.markdown("---")

# 4. CHARGEMENT MODÈLE
# ==========================================
POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiZmIxZTcxZjEtYjllNi00ODY0LTkxOWItNGYyNzMzZGMwMGNlIiwidCI6ImQwMzAyZmNjLTNlODEtNDljMy04MjM1LWQzMTFhMzY4NGNmYyJ9"


@st.cache_resource
def load_ml_artifacts_local():
    model_path = "churn_model.pkl"
    if os.path.exists(model_path):
        try:
            artifacts = joblib.load(model_path)
            return artifacts.get("model"), artifacts.get("features")
        except Exception as e:
            st.error(f"Erreur de chargement du modèle: {e}")
            return None, None
    return None, None


model, model_features = load_ml_artifacts_local()


if st.session_state["active_tab"] == "dashboard":

    st.components.v1.iframe(
        POWERBI_URL,
        height=800,
        scrolling=True
    )

# --- ONGLET 2 : PRÉDICTION ML ---

elif st.session_state["active_tab"] == "prediction":

    st.caption(
        "Remplissez les informations du client ou utilisez un profil type."
    )

    demo_col1, demo_col2, _ = st.columns([1, 1, 2])
    preset = None
    with demo_col1:
        if st.button("Profil A (Fidèle)", use_container_width=True):
            preset = "Profil_A"
    with demo_col2:
        if st.button("Profil B (Risque)", use_container_width=True):
            preset = "Profil_B"

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

    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        with col1:
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
                "Jours depuis l'inscription", min_value=1, value=def_tenure
            )
            customer_span_days = st.number_input(
                "Durée d'activité (jours)", min_value=0, value=def_span
            )

        with col2:
            total_spent = st.number_input(
                "Montant Total Dépensé ($)",
                min_value=0.0,
                value=float(def_spent),
            )
            frequency = st.number_input(
                "Nombre de commandes", min_value=1, value=def_freq
            )
            total_quantity = st.number_input(
                "Quantité d'articles", min_value=1, value=def_qty
            )
            unique_products = st.number_input(
                "Produits uniques", min_value=1, value=def_uniq
            )

        submit_btn = st.form_submit_button(
            "🚀 Calculer le risque", use_container_width=True
        )

    if submit_btn:
        if model is not None and model_features is not None:
            gender_encoded = 1 if gender_str == "Homme" else 0
            online_encoded = 1 if is_online_shopper else 0

            avg_basket = total_spent / frequency if frequency > 0 else 0
            purchase_rate = frequency / (tenure_days + 1)
            avg_item_price = (
                total_spent / total_quantity if total_quantity > 0 else 0
            )
            monetary_velocity = total_spent / (tenure_days + 1)
            basket_depth = unique_products / (frequency + 1)

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

            input_df = pd.DataFrame([input_dict])[model_features]
            probability = model.predict_proba(input_df)[0][1] * 100

            st.divider()
            if probability >= 50:
                st.error(
                    f"⚠️ **Risque Élevé de Churn : {probability:.1f}%** (Déclencher campagne de ré-engagement)"
                )
            else:
                st.success(
                    f"✅ **Client Fidèle : Risque de Churn {probability:.1f}%**"
                )
        else:
            st.error(" Modèle de prédiction introuvable.")