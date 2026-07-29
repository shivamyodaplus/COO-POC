import streamlit as st

st.set_page_config(
    page_title="Template Retrieval",
    page_icon="🔍",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/upload.py",   title="Upload Template",   icon="📤"),
    st.Page("pages/retrieve.py", title="Retrieve Template", icon="🔎"),
])
pg.run()
