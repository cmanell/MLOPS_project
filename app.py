import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Prédiction de défaut de crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("models/best_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6f8fc 0%, #eef3ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }
    .hero-card {
        background: linear-gradient(135deg, #163b87 0%, #2457c5 100%);
        padding: 2rem 2.2rem;
        border-radius: 24px;
        color: white;
        box-shadow: 0 18px 45px rgba(28, 63, 143, 0.18);
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-text {
        font-size: 1rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    .soft-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(29, 78, 216, 0.08);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        text-align: center;
    }
    .metric-label {
        color: #5b6475;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        color: #163b87;
        font-size: 1.35rem;
        font-weight: 800;
    }
    .result-good {
        background: linear-gradient(135deg, #e8fff3 0%, #dff8e8 100%);
        border: 1px solid #b8ebc9;
        color: #106a37;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        font-weight: 700;
    }
    .result-mid {
        background: linear-gradient(135deg, #fff7e8 0%, #fff2d8 100%);
        border: 1px solid #f0d493;
        color: #9a6500;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        font-weight: 700;
    }
    .result-high {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe2e2 100%);
        border: 1px solid #f0b1b1;
        color: #a32121;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        font-weight: 700;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #163b87 0%, #2457c5 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(36, 87, 197, 0.25);
    }
    div.stButton > button:hover {
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Charge le modèle et le scaler."""
    if not MODEL_PATH.exists():
        st.error("Modèle introuvable. Vérifie le fichier best_model.joblib")
        st.stop()

    if not SCALER_PATH.exists():
        st.error("Scaler introuvable. Vérifie le fichier scaler.joblib")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def score_color(probability: float) -> str:
    if probability < 0.30:
        return "#18a058"
    if probability < 0.70:
        return "#d18b00"
    return "#d03050"


def result_box(probability: float, prediction: int) -> str:
    if probability < 0.30:
        return f'<div class="result-good">✅ Risque faible de défaut — probabilité estimée : {probability:.2%}</div>'
    if probability < 0.70:
        return f'<div class="result-mid">⚠️ Risque modéré — probabilité estimée : {probability:.2%}</div>'
    return f'<div class="result-high">🚨 Risque élevé de défaut — probabilité estimée : {probability:.2%}</div>'


def main():
    model, scaler = load_artifacts()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">💳 Prédiction de défaut de crédit</div>
            <div class="hero-text">
                Cette application estime la probabilité qu’un client fasse défaut sur un prêt personnel à partir
                de ses caractéristiques financières et professionnelles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2, top3 = st.columns(3)
    with top1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Modèle</div><div class="metric-value">Classification</div></div>',
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Cible</div><div class="metric-value">Défaut client</div></div>',
            unsafe_allow_html=True,
        )
    with top3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Sortie</div><div class="metric-value">Probabilité</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Informations du client")
        c1, c2 = st.columns(2)

        with c1:
            credit_lines_outstanding = st.number_input(
                "Nombre de lignes de crédit",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
            )
            loan_amt_outstanding = st.number_input(
                "Montant du prêt restant",
                min_value=0.0,
                value=3000.0,
                step=100.0,
            )
            total_debt_outstanding = st.number_input(
                "Dette totale restante",
                min_value=0.0,
                value=2500.0,
                step=100.0,
            )

        with c2:
            income = st.number_input(
                "Revenu annuel",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )
            years_employed = st.number_input(
                "Ancienneté professionnelle (années)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
            )
            fico_score = st.number_input(
                "Score FICO",
                min_value=300,
                max_value=850,
                value=620,
                step=1,
            )

        predict = st.button("Prédire le risque", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Aide à la lecture")
        st.markdown(
            """
            - **Risque faible** : probabilité < 30%
            - **Risque modéré** : entre 30% et 70%
            - **Risque élevé** : probabilité > 70%

            L’application applique le même **scaler** que celui utilisé pendant l’entraînement du modèle.
            """
        )
        with st.expander("Variables utilisées"):
            st.write(
                [
                    "credit_lines_outstanding",
                    "loan_amt_outstanding",
                    "total_debt_outstanding",
                    "income",
                    "years_employed",
                    "fico_score",
                ]
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if predict:
        features = np.array(
            [[
                credit_lines_outstanding,
                loan_amt_outstanding,
                total_debt_outstanding,
                income,
                years_employed,
                fico_score,
            ]],
            dtype=float,
        )

        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        proba = float(model.predict_proba(features_scaled)[0, 1])

        st.write("")
        res1, res2 = st.columns([1.1, 0.9], gap="large")

        with res1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Résultat de la prédiction")
            st.markdown(result_box(proba, prediction), unsafe_allow_html=True)
            st.write("")
            st.progress(proba)
            st.caption(f"Probabilité de défaut estimée : {proba:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with res2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Interprétation")
            if proba < 0.30:
                st.write("Le client présente un faible niveau de risque. Le profil semble globalement rassurant.")
            elif proba < 0.70:
                st.write("Le client présente un niveau de risque intermédiaire. Une analyse complémentaire peut être utile.")
            else:
                st.write("Le client présente un risque élevé de défaut. Une vigilance renforcée est recommandée.")

            st.markdown(
                f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-label">Score de risque</div>
                    <div class="metric-value" style="color:{score_color(proba)};">{proba:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.caption("Projet MLOps — Université Sorbonne | Application de démonstration Streamlit")


if __name__ == "__main__":
    main()
