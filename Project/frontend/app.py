import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Dashboard", page_icon="🚀")
st.title("Dashboard")
st.caption(f"Backend: `{BACKEND_URL}`")

st.divider()

if st.button("Check Backend Status", type="primary"):
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/status", timeout=5)
        r.raise_for_status()
        st.success("Backend is reachable")
        st.json(r.json())
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend.")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
