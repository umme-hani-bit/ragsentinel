import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("RAGsentinel")
st.write("Tiny UI wired to the API.")

try:
    r = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
    st.success(f"API says: {r.json()}")
except Exception as e:
    st.error(f"API not reachable: {e}")