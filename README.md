
# Bankruptcy Prediction App

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the app:
   ```
   streamlit run app.py
   ```

3. Open in browser: http://localhost:8501

## Input
- Upload a `.csv` file with these columns:
  industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk

- Or enter values manually in sidebar.

## Output
- Prediction: Bankruptcy / Non-Bankruptcy
- Probability of Bankruptcy
- Suggested Action
- EDA Insights

## Notes
⚠️ Place your trained `logistic_model.pkl` and `bankrupt_clean.csv` in the same folder as app.py.
