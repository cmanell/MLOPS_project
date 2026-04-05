import joblib
import numpy as np
import streamlit as st
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="Prédiction de défaut de crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("models/logreg_smote.joblib")


# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(61, 99, 255, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
            linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
        color: white;
        border-radius: 28px;
        padding: 2rem 2.2rem;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.2rem;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 850;
        line-height: 1.08;
        margin-bottom: 0.55rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.65;
        color: rgba(255,255,255,0.92);
        max-width: 760px;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
    }

    .stat-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .panel {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.2rem 1.25rem 1.1rem 1.25rem;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
    }

    .panel-title {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .panel-subtitle {
        color: #64748b;
        font-size: 0.94rem;
        margin-bottom: 1rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #0ea5e9 100%);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
    }

    div.stButton > button:hover {
        color: white;
        border: none;
    }

    .risk-card {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.85rem;
    }

    .risk-low {
        background: linear-gradient(135deg, #ebfff3 0%, #dcfce7 100%);
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .risk-mid {
        background: linear-gradient(135deg, #fff9e9 0%, #fef3c7 100%);
        color: #92400e;
        border: 1px solid #fde68a;
    }

    .risk-high {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    .score-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        margin-top: 0.8rem;
    }

    .score-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }

    .score-value {
        font-size: 1.8rem;
        font-weight: 850;
        letter-spacing: -0.02em;
    }

    .section-gap {
        height: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Modèle introuvable. Vérifie la présence de models/best_model.joblib")
        st.stop()
    return joblib.load(MODEL_PATH)


# =========================
# HELPERS
# =========================
def get_risk_level(probability: float):
    if probability < 0.30:
        return "Faible", "risk-low", "✅ Risque faible de défaut"
    if probability < 0.70:
        return "Modéré", "risk-mid", "⚠️ Risque modéré de défaut"
    return "Élevé", "risk-high", "🚨 Risque élevé de défaut"


def get_score_color(probability: float) -> str:
    if probability < 0.30:
        return "#16a34a"
    if probability < 0.70:
        return "#d97706"
    return "#e11d48"


# =========================
# APP
# =========================
def main():
    model = load_model()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Projet MLOps • Crédit bancaire</div>
            <div class="hero-title">Prédiction de défaut de crédit</div>
            <div class="hero-subtitle">
                Cette application estime la probabilité qu’un client fasse défaut sur un prêt personnel à partir
                de ses caractéristiques financières. L’objectif est d’aider à une prise de décision plus rapide,
                plus cohérente et plus interprétable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Type de modèle</div>
                <div class="stat-value">Classification binaire</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Objectif</div>
                <div class="stat-value">Estimer le risque client</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sortie</div>
                <div class="stat-value">Probabilité de défaut</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 0.95], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Informations du client</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Renseignez les variables utilisées par le modèle pour générer la prédiction.</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="medium")

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
                max_value=100000.0,
                value=3000.0,
                step=100.0,
            )
            total_debt_outstanding = st.number_input(
                "Dette totale restante",
                min_value=0.0,
                max_value=100000.0,
                value=2500.0,
                step=100.0,
            )

        with c2:
            income = st.number_input(
                "Revenu annuel",
                min_value=0.0,
                max_value=300000.0,
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

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        predict = st.button("Prédire le risque", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Guide de lecture</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Interprétez rapidement le score retourné par le modèle.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **Risque faible** : probabilité inférieure à **30 %**
            - **Risque modéré** : probabilité entre **30 % et 70 %**
            - **Risque élevé** : probabilité supérieure à **70 %**
            """
        )
        st.info("Le pipeline sauvegardé applique automatiquement le même prétraitement que pendant l’entraînement.")

        with st.expander("Variables utilisées par le modèle"):
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

        prediction = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0, 1])
        risk_level, risk_class, risk_text = get_risk_level(proba)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        res1, res2 = st.columns([1.1, 0.9], gap="large")

        with res1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Résultat de la prédiction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-card {risk_class}">{risk_text} — probabilité estimée : {proba:.2%}</div>', unsafe_allow_html=True)
            st.progress(proba)
            st.caption(f"Sortie du modèle : {proba:.2%} de probabilité de défaut")
            st.markdown('</div>', unsafe_allow_html=True)

        with res2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Interprétation</div>', unsafe_allow_html=True)
            if proba < 0.30:
                st.write("Le profil semble globalement rassurant. Le risque de défaut estimé reste faible.")
            elif proba < 0.70:
                st.write("Le profil présente un niveau de risque intermédiaire. Une analyse complémentaire peut être pertinente.")
            else:
                st.write("Le profil présente un niveau de risque élevé. Une vigilance renforcée est recommandée avant décision.")

            st.markdown(
                f"""
                <div class="score-box">
                    <div class="score-label">Niveau de risque</div>
                    <div class="score-value" style="color:{get_score_color(proba)};">{risk_level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Projet MLOps — Université Sorbonne | Application de démonstration Streamlit")


if __name__ == "__main__":
    main()
