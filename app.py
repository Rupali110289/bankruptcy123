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
        .dataframe th, .dataframe td {
            white-space: nowrap;
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
        eda_df['class_yn'] = pred_temp.astype(int)
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

                # Fix for EDA
                eda_df = df.copy()
                eda_df['class_yn'] = pred.astype(int)
                st.session_state.eda_df = eda_df

                # widen columns
                st.markdown("""
                    <style>
                    .css-1d391kg td {
                        max-width: 200px;
                        word-wrap: break-word;
                    }
                    </style>
                """, unsafe_allow_html=True)

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
    st.markdown("## EDA Insights (Uploaded Data)")
    st.write("""
    These insights help you understand how different risk factors relate to bankruptcy
    in the **uploaded data**. Charts are interactive — hover to see details.
    """)

    eda_df = st.session_state.eda_df.copy()

    # Ensure class_yn is present and mapped
    if 'class_yn' not in eda_df.columns:
        pred, _ = predict_risk(eda_df[REQUIRED_COLS])
        eda_df['class_yn'] = pred

    eda_df['Outcome'] = eda_df['class_yn'].map({0: "Bankruptcy", 1: "Non-Bankruptcy"})

    eda_option = st.radio(
        "Choose an insight to explore:",
        [
            "Industrial Risk vs Bankruptcy",
            "Financial Flexibility vs Bankruptcy",
            "Key Highlights"
        ]
    )

    if eda_option == "Industrial Risk vs Bankruptcy":
        st.markdown("""
        Higher industrial risk often leads to a higher chance of bankruptcy.  
        This chart shows the percentage of companies in each industrial risk level that went bankrupt or not.
        """)
        perc_df = eda_df.groupby(['industrial_risk', 'Outcome']).size().reset_index(name='count')
        perc_df['percent'] = perc_df.groupby('industrial_risk')['count'].transform(lambda x: x/x.sum()*100)

        fig = px.bar(
            perc_df,
            x='industrial_risk', y='percent', color='Outcome',
            barmode='group', text=perc_df['percent'].round(1),
            title="Industrial Risk vs Bankruptcy (%)",
            labels={'industrial_risk': 'Industrial Risk Level', 'percent': 'Percentage', 'Outcome': 'Outcome'},
            color_discrete_map={"Bankruptcy": 'red', "Non-Bankruptcy": 'green'},
            category_orders={"industrial_risk": [0, 0.5, 1], "Outcome": ["Bankruptcy", "Non-Bankruptcy"]}
        )
        fig.update_layout(xaxis_title="Industrial Risk", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Financial Flexibility vs Bankruptcy":
        st.markdown("""
        Financial flexibility reflects a company’s ability to deal with financial stress.  
        Higher flexibility usually reduces bankruptcy risk.
        """)
        perc_df = eda_df.groupby(['financial_flexibility', 'Outcome']).size().reset_index(name='count')
        perc_df['percent'] = perc_df.groupby('financial_flexibility')['count'].transform(lambda x: x/x.sum()*100)

        fig = px.bar(
            perc_df,
            x='financial_flexibility', y='percent', color='Outcome',
            barmode='group', text=perc_df['percent'].round(1),
            title="Financial Flexibility vs Bankruptcy (%)",
            labels={'financial_flexibility': 'Financial Flexibility Level', 'percent': 'Percentage', 'Outcome': 'Outcome'},
            color_discrete_map={"Bankruptcy": 'red', "Non-Bankruptcy": 'green'},
            category_orders={"financial_flexibility": [0, 0.5, 1], "Outcome": ["Bankruptcy", "Non-Bankruptcy"]}
        )
        fig.update_layout(xaxis_title="Financial Flexibility", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("### Key Highlights from Uploaded Data")

        bankrupt_rate = (eda_df['class_yn'] == 0).mean() * 100
        high_risk_bankrupt = eda_df.query("industrial_risk == 1 & class_yn == 0").shape[0]
        low_flex_bankrupt = eda_df.query("financial_flexibility == 0 & class_yn == 0").shape[0]
        high_operating_bankrupt = eda_df.query("operating_risk == 1 & class_yn == 0").shape[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Bankruptcy Rate", f"{bankrupt_rate:.1f} %")
        col2.metric("High Industrial Risk & Bankrupt", high_risk_bankrupt)
        col3.metric("Low Financial Flexibility & Bankrupt", low_flex_bankrupt)
        col4.metric("High Operating Risk & Bankrupt", high_operating_bankrupt)

        st.write("---")

        if st.checkbox("Show Data Table"):
            st.dataframe(eda_df.head(20))

# ========== FEATURE IMPORTANCE ==========
with tabs[3]:
    st.markdown("## 📈 Feature Importance (on Uploaded Data)")

    eda_df = st.session_state.eda_df.copy()
    eda_clean = eda_df[REQUIRED_COLS]
    pred, proba = predict_risk(eda_clean)

    # Add predictions & probability
    eda_clean = eda_clean.copy()
    eda_clean["Predicted_Class"] = np.where(pred == 0, "Bankruptcy", "Non-Bankruptcy")
    eda_clean["Bankruptcy_Probability"] = proba

    st.markdown(
        """
        This section helps you understand **which features have the most influence** on bankruptcy probability 
        based on your uploaded data.
        
        🔷 Higher correlation means that the feature strongly impacts the risk.  
        🔷 Blue bars below show the relative importance of each factor.
        """
    )

    # Compute correlation (absolute) of each feature with bankruptcy probability
    corr = eda_clean[REQUIRED_COLS + ["Bankruptcy_Probability"]].corr()["Bankruptcy_Probability"]\
        .drop("Bankruptcy_Probability").abs().sort_values(ascending=False) * 100

    # Interactive bar chart
    fig = px.bar(
        x=corr.index,
        y=corr.values.round(2),
        title="📊 Feature Importance: Correlation with Bankruptcy Probability",
        labels={"x": "Feature", "y": "|Correlation with Bankruptcy Probability| (%)"},
        color=corr.values,
        color_continuous_scale="Blues",
        text=corr.values.round(1)
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Absolute Correlation (%)",
        xaxis_tickangle=-45,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)

    most_impactful = corr.idxmax()
    impact_value = corr.max()
    st.success(
        f"🎯 The most influential feature (on your uploaded data) is **{most_impactful}**, "
        f"with an estimated **{impact_value:.2f}% correlation** to the predicted bankruptcy probability."
    )

    # Optional: show table
    with st.expander("📋 View Full Correlation Table & Explanation"):
        st.markdown(
            """
            Below is the full table showing how each feature correlates (in %) with the probability of bankruptcy.
            The higher the value, the more it impacts the prediction.
            """
        )
        st.table(
            corr.round(2).reset_index()
            .rename(columns={"index": "Feature", "Bankruptcy_Probability": "Correlation (%)"})
        )

        st.info("""
        💡 **How to interpret?**
        - Features with higher correlation values have a stronger impact on the bankruptcy prediction.
        - Positive or negative correlation does not matter here — we look at the absolute strength of the relationship.
        - Example: If *competitiveness* shows ~80%, it means competitiveness strongly affects the bankruptcy probability in your data.
        """)
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

    # Add the Final Loan Decision section here
    st.write("---")
    st.markdown("## Final Loan Decision")

    st.markdown(f"""
    **High Risk (≥70%)**: {high} companies  
    → **Recommendation**: DECLINE LOAN  
    → **Why**: Bankruptcy probability above 70%, too risky.

    **Medium Risk (50–70%)**: {medium} companies  
    → **Recommendation**: APPROVE WITH COLLATERAL  
    → **Why**: Moderate risk, approve only with sufficient collateral.

    **Low Risk (<50%)**: {low} companies  
    → **Recommendation**: APPROVE LOAN  
    → **Why**: Low risk of bankruptcy, safe to lend.

    **Overall Average Risk**: {avg:.2f}%  
    """)

    if avg < 50 and low >= (total/2):
        st.success("Final Decision: YES — Offer loans to Low Risk companies, with caution on Medium Risk.")
    else:
        st.warning("Final Decision: BE CAUTIOUS — High proportion of risky companies detected.")

# ========== LOGOUT ==========
if st.button("Logout", key="logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()
