import streamlit as st

st.set_page_config(
    page_title="Document Template Intelligence",
    page_icon="🔍",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/upload.py", title="Upload Document", icon="📤"),
    st.Page("pages/retrieve.py", title="Retrieve Template", icon="🔎"),
    st.Page("pages/visual_templates.py", title="Visual Templates", icon="✍️"),
])
pg.run()
