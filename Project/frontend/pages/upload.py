from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Upload Template")
st.caption("Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP")

with st.form("upload_form"):
    uploaded = st.file_uploader(
        "Select document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
    )
    country  = st.text_input("Country",       placeholder="e.g. Egypt")
    doc_type = st.text_input("Document Type", placeholder="e.g. CoO")
    submitted = st.form_submit_button("Upload & Index", type="primary")

if submitted:
    if not uploaded:
        st.error("Please select a file.")
    elif not country.strip():
        st.error("Country is required.")
    elif not doc_type.strip():
        st.error("Document Type is required.")
    else:
        with st.spinner("Preprocessing, embedding and indexing..."):
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/documents/upload",
                data={"country": country.strip(), "doc_type": doc_type.strip()},
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=300,
            )
        if resp.ok:
            data = resp.json()
            st.success(
                f"Indexed **{data['pages_stored']}** page(s)  \n"
                f"ID: `{data['id']}`  \n"
                f"Country: **{data['country']}** | Type: **{data['doc_type']}**"
            )
        else:
            st.error(f"Upload failed ({resp.status_code}): {resp.text}")
