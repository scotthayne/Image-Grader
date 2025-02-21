import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Modal Streamlit Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Modal Streamlit Demo")

# Generate sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=100),
    'Value': np.random.randn(100).cumsum()
})

# Sidebar controls
st.sidebar.header("Dashboard Controls")
window = st.sidebar.slider("Moving Average Window", 1, 50, 10)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data")
    st.line_chart(data.set_index('Date')['Value'])

with col2:
    st.subheader(f"{window}-Day Moving Average")
    ma_data = data.copy()
    ma_data['MA'] = ma_data['Value'].rolling(window=window).mean()
    st.line_chart(ma_data.set_index('Date')['MA'])

st.dataframe(data.tail(), use_container_width=True)