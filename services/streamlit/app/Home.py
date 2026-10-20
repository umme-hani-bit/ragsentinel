import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAGsentinel", page_icon="🛡️")
st.title("RAGsentinel")
st.write("Hello! This is the placeholder dashboard.")

try:
    r = requests.get(f"{API_BASE_URL}/healthz", timeout=3)
    st.success(f"API health: {r.json()}")
except Exception as e:
    st.warning(f"API not reachable yet: {e}")