import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os

# ✅ First Streamlit command
st.set_page_config(page_title="📈 Stock Price Prediction App", layout="centered")

# ✅ Set Background + Button Style
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Fix button color */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #FF1E1E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Models with error handling
try:
    model_reliance = joblib.load(os.path.join('models', 'reliance_stock.pkl'))
    model_nhpc = joblib.load(os.path.join('models', 'nhpc_stock.pkl'))
    model_adani = joblib.load(os.path.join('models', 'adani_stock.pkl'))
except Exception as e:
    st.error(f"Error loading models: {e}")

company_model = {
    "Reliance Industries": model_reliance,
    "NHPC Ltd": model_nhpc,
    "Adani Enterprises": model_adani
}

# ✅ UI
st.title("📈 Stock Price Prediction App")

company = st.selectbox(
    "Select a Company",
    ("Reliance Industries", "NHPC Ltd", "Adani Enterprises")
)

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# ✅ Date validation
if start_date > end_date:
    st.error("Start Date cannot be later than End Date.")

# ✅ Predict button
if st.button('Predict'):
    st.success(f"Generating prediction for {company} from {start_date} to {end_date}")

    model = company_model[company]

    # 🔥 Example dummy data (replace later with real prediction logic)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    actual_prices = [100 + i for i in range(len(dates))]          # Simulated real prices
    predicted_prices = [price + 2 for price in actual_prices]     # Simulated prediction

    # 📊 Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=dates, y=predicted_prices, mode='lines', name='Predicted Price'))

    fig.update_layout(
        title=f"{company} Stock Price Prediction 📈",
        xaxis_title="Date",
        yaxis_title="Stock Price (INR)",
        legend_title="Legend",
        template="plotly_dark",
        hovermode="x unified"
    )

    # Display Plotly chart
    st.plotly_chart(fig, use_container_width=True)
