import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #2C3E50;
            color: white;
            border: None;
            padding: 8px 20px;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #1A242F;
        }
        .big-font {
            font-size:18px !important;
            font-weight:500;
        }
        .metric {
            font-weight:bold;
            color:#34495E;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== LOGIN LOGIC ==========
VALID_USERS = {
    "admin": "password123",
    "Banker1": "sbi123@",
    "Banker2": "icici456!",
    "Manager": "manager789"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'eda_df' not in st.session_state:
    st.session_state.eda_df = pd.read_csv("bankrupt_clean.csv")

REQUIRED_COLS = [
    'industrial_risk', 'management_risk',
    'financial_flexibility', 'credibility',
    'competitiveness', 'operating_risk'
]

# ========== LOAD MODEL & SCALER ==========
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
bankruptcy_class_idx = model.classes_.tolist().index(0)

# ========== HELPERS ==========
def predict_risk(data):
    data_scaled = scaler.transform(data[REQUIRED_COLS])
    pred = model.predict(data_scaled)
    proba = model.predict_proba(data_scaled)[:, bankruptcy_class_idx]
    return pred, proba

def get_suggestion(prob):
    if prob >= 0.7:
        return "Decline Loan"
    elif prob >= 0.5:
        return "Approve with Collateral"
    else:
        return "Approve"

def feature_importance():
    coef = pd.Series(model.coef_[0], index=REQUIRED_COLS)
    return coef

def example_csv():
    return pd.DataFrame({col: [0] for col in REQUIRED_COLS})

# ========== ENSURE class_yn in eda_df ==========
eda_df = st.session_state.eda_df
if 'class_yn' not in eda_df.columns:
    if 'class' in eda_df.columns:
        eda_df['class_yn'] = eda_df['class'].map({'non-bankruptcy': 1, 'bankruptcy': 0})
    else:
        X_temp = eda_df[REQUIRED_COLS]
        pred_temp, _ = predict_risk(X_temp)
        eda_df['class_yn'] = pred_temp
st.session_state.eda_df = eda_df

# ========== LOGIN ==========
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center;'>🔒 Login to Access the Dashboard</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in VALID_USERS and password == VALID_USERS[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# ========== HEADER ==========
st.markdown(f"<h1 style='color:#2C3E50;'>📊 Bankruptcy Prediction Dashboard</h1>", unsafe_allow_html=True)

# ========== TABS ==========
tabs = st.tabs(["Overview", "Prediction", "EDA", "Importance", "Summary"])

# ========== OVERVIEW ==========
with tabs[0]:
    st.markdown("## About this App")
    st.markdown("""
This dashboard predicts the **probability of bankruptcy** for companies based on six risk factors.

### Features:
- Upload CSV or enter data manually
- Get predictions & recommendations
- Explore EDA insights
- View feature importance

### Decision Rules:
- ≥ 70% → Decline Loan
- 50–70% → Approve with Collateral
- < 50% → Approve
    """)
    st.download_button(
        "Download Example CSV",
        example_csv().to_csv(index=False),
        "example.csv"
    )

# ========== PREDICTION ==========
with tabs[1]:
    st.markdown("## Prediction")
    col1, col2 = st.columns([1, 3])

    with col1:
        input_mode = st.radio("Select Input Mode", ["Upload CSV", "Manual Entry"])

        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        else:
            single = {col: st.selectbox(col, [0, 0.5, 1]) for col in REQUIRED_COLS}

    with col2:
        if input_mode == "Upload CSV":
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                missing = [col for col in REQUIRED_COLS if col not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                    st.stop()

                df = df[REQUIRED_COLS]

                pred, proba = predict_risk(df)

                df['Prediction'] = np.where(pred == 0, "Bankruptcy", "Non-Bankruptcy")
                df['Probability of Bankruptcy (%)'] = (proba * 100).round(2)
                df['Loan Suggestion'] = df['Probability of Bankruptcy (%)'].apply(lambda x: get_suggestion(x / 100))

                eda_df = df.copy()
                eda_df['class_yn'] = pred
                st.session_state.eda_df = eda_df

                st.markdown("### Prediction Results")
                st.dataframe(df.style.background_gradient(cmap="Blues", axis=0))
                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

        else:
            single_df = pd.DataFrame([single])
            pred, proba = predict_risk(single_df)

            prediction_text = "Bankruptcy" if pred[0] == 0 else "Non-Bankruptcy"
            suggestion_text = get_suggestion(proba[0])

            levels = {0: "Low", 0.5: "Medium", 1: "High"}
            report_df = pd.DataFrame({
                "Feature": single_df.columns,
                "Value": single_df.iloc[0].values,
                "Risk Level": [levels[v] for v in single_df.iloc[0].values]
            })

            st.table(report_df)
            st.markdown(f"**Prediction:** {prediction_text}")
            st.markdown(f"**Probability of Bankruptcy:** {proba[0]*100:.2f}%")
            st.markdown(f"**Suggestion:** {suggestion_text}")

# ========== EDA ==========
with tabs[2]:
    st.markdown("## EDA Insights")
    eda_df = st.session_state.eda_df
    eda_option = st.selectbox("Select EDA Insight", [
        "Industrial Risk % by Class",
        "Financial Flexibility % by Class",
        "Top 3 Insights"
    ])

    if eda_option == "Industrial Risk % by Class":
        perc_df = eda_df.groupby(['industrial_risk', 'class_yn']).size().reset_index(name='count')
        perc_df['percent'] = perc_df.groupby('industrial_risk')['count'].transform(lambda x: x/x.sum()*100)

        fig = px.bar(perc_df, x='industrial_risk', y='percent', color='class_yn',
                     barmode='group', text='percent',
                     title="Industrial Risk vs Bankruptcy (%)",
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Financial Flexibility % by Class":
        perc_df = eda_df.groupby(['financial_flexibility', 'class_yn']).size().reset_index(name='count')
        perc_df['percent'] = perc_df.groupby('financial_flexibility')['count'].transform(lambda x: x/x.sum()*100)

        fig = px.bar(perc_df, x='financial_flexibility', y='percent', color='class_yn',
                     barmode='group', text='percent',
                     title="Financial Flexibility vs Bankruptcy (%)",
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("""
- Companies with high industrial risk → ~80% chance of bankruptcy.
- High financial flexibility lowers risk significantly.
- High operating risk observed in ~70% of bankrupt companies.
""")

# ========== FEATURE IMPORTANCE ==========
with tabs[3]:
    st.markdown("## Feature Importance")
    coef = feature_importance()
    coef_pct = (coef / coef.abs().sum()) * 100

    fig = px.bar(coef_pct, x=coef_pct.index, y=coef_pct.values,
                 title="Feature Importance (% Contribution)",
                 labels={"y": "% Contribution", "x": "Feature"},
                 color=coef_pct.values, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    most_impactful = coef.abs().idxmax()
    impact_value = coef[most_impactful]
    st.info(f"Most influential feature: **{most_impactful}**, contributing about {impact_value:.2f} units.")

# ========== SUMMARY METRICS ==========
with tabs[4]:
    st.markdown("## Summary Metrics")
    eda_df = st.session_state.eda_df
    total = len(eda_df)
    eda_clean = eda_df[REQUIRED_COLS]
    pred, proba = predict_risk(eda_clean)

    high = sum(proba >= 0.7)
    medium = sum((proba >= 0.5) & (proba < 0.7))
    low = sum(proba < 0.5)
    avg = np.mean(proba) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("High Risk (≥70%)", high)
    col3.metric("Medium Risk (50–70%)", medium)
    col4.metric("Low Risk (<50%)", low)
    col5.metric("Avg Risk %", f"{avg:.2f}%")

    fig = px.pie(
        names=["High Risk", "Medium Risk", "Low Risk"],
        values=[high, medium, low],
        title="Risk Distribution",
        color_discrete_sequence=["#8E44AD", "#E67E22", "#27AE60"]

    )
    st.plotly_chart(fig, use_container_width=True)

# ========== LOGOUT ==========
if st.button("Logout", key="logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

