import streamlit as st

st.write("Secret keys in the system:")
for key in st.secrets:
    st.write(f"Found key: {key}")

st.write("Trying to access AssemblyAI API key:")
try:
    key = st.secrets["ASSEMBLYAI_API_KEY"]
    st.success(f"Successfully found AssemblyAI API key: {key[:4]}...")
except Exception as e:
    st.error(f"Error accessing key: {str(e)}")