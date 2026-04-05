import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Prédiction de défaut de crédit", page_icon="💳", layout="centered")

MODEL_PATH = Path("models/best_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")


@st.cache_resource
def load_artifacts():
    """Charge le modèle et le scaler"""
    if not MODEL_PATH.exists():
        st.error("Modèle introuvable. Vérifie le fichier best_model.joblib")
        st.stop()

    if not SCALER_PATH.exists():
        st.error("Scaler introuvable. Vérifie le fichier scaler.joblib")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def main():
    st.title("💳 Prédiction de défaut de crédit")
    st.write("Estimez la probabilité qu’un client fasse défaut sur un prêt.")

    model, scaler = load_artifacts()

    st.subheader("Informations du client")

    col1, col2 = st.columns(2)

    with col1:
        credit_lines_outstanding = st.number_input(
            "Nombre de lignes de crédit",
            min_value=0,
            max_value=20,
            value=1
        )

        loan_amt_outstanding = st.number_input(
            "Montant du prêt",
            min_value=0.0,
            value=3000.0
        )

        total_debt_outstanding = st.number_input(
            "Dette totale",
            min_value=0.0,
            value=2500.0
        )

    with col2:
        income = st.number_input(
            "Revenu annuel",
            min_value=0.0,
            value=50000.0
        )

        years_employed = st.number_input(
            "Années d'emploi",
            min_value=0,
            max_value=50,
            value=5
        )

        fico_score = st.number_input(
            "Score FICO",
            min_value=300,
            max_value=850,
            value=620
        )

    if st.button("Prédire le risque"):
        features = np.array([[
            credit_lines_outstanding,
            loan_amt_outstanding,
            total_debt_outstanding,
            income,
            years_employed,
            fico_score
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0, 1]

        st.subheader("Résultat")

        if prediction == 1:
            st.error(f"⚠️ Risque élevé de défaut : {proba:.2%}")
        else:
            st.success(f"✅ Risque faible de défaut : {proba:.2%}")

        st.progress(float(proba))

        st.markdown("### Interprétation")
        if proba < 0.30:
            st.write("Le client présente un faible risque.")
        elif proba < 0.70:
            st.write("Le client présente un risque modéré.")
        else:
            st.write("Le client présente un risque élevé.")

    st.markdown("---")
    st.caption("Projet MLOps — Université Sorbonne")


if __name__ == "__main__":
    main()