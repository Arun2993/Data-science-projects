import streamlit as st
import pandas as pd

st.title("🛒 E-Commerce RFM Analysis App")

# Upload data
uploaded_file = st.file_uploader(r"C:\Users\Krishi\Documents\Learnbay\Project class\E-Commerce Domain-20250419T181517Z-001\E-Commerce Domain\RFMApproach.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("🔍 Data Preview")
    st.write(df.head())

    st.subheader("📊 RFM Metrics Summary")
    st.write(df[['Recency', 'Frequency', 'Monetary']].describe())

    st.subheader("🎯 RFM Segmentation Counts")
    st.write(df['RFM'].value_counts())

    # Bar Charts
    st.subheader("📈 RFM Metrics Distribution")
    st.bar_chart(df[['Recency', 'Frequency', 'Monetary']])

    st.subheader("💾 Download Processed RFM Data")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download RFM Data as CSV",
        data=csv,
        file_name='rfm_processed.csv',
        mime='text/csv',
    )

else:
    st.info("👆 Please upload your RFM CSV file to start.")

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app performs analysis on pre-calculated RFM scores.
Upload your RFM file to view insights and download updated data.
Developed with ❤️ using Streamlit.
""")
