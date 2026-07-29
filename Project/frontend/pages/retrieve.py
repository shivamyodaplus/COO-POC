from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Retrieve Template")
st.caption("Upload a query document to find the closest matching template.")

with st.form("retrieve_form"):
    uploaded = st.file_uploader(
        "Query document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
    )
    col1, col2 = st.columns(2)
    with col1:
        country  = st.text_input("Country filter (optional)",       placeholder="e.g. Egypt")
    with col2:
        doc_type = st.text_input("Document Type filter (optional)", placeholder="e.g. CoO")
    top_k     = st.slider("Top-K results", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not uploaded:
        st.error("Please upload a query document.")
    else:
        with st.spinner("Running hybrid search (dense + sparse)..."):
            form_data = {"top_k": str(top_k)}
            if country.strip():
                form_data["country"]  = country.strip()
            if doc_type.strip():
                form_data["doc_type"] = doc_type.strip()

            resp = requests.post(
                f"{BACKEND_URL}/api/v1/documents/retrieve",
                data=form_data,
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=300,
            )

        if not resp.ok:
            st.error(f"Search failed ({resp.status_code}): {resp.text}")
        else:
            results = resp.json()
            if not results:
                st.warning("No matching templates found.")
            else:
                st.success(f"Found {len(results)} result(s)")
                st.divider()

                # Show query image on the left, results on the right
                query_img = None
                try:
                    query_img = Image.open(io.BytesIO(uploaded.getvalue()))
                except Exception:
                    pass

                for rank, hit in enumerate(results, start=1):
                    with st.container():
                        c_query, c_match, c_info = st.columns([1, 1, 2])
                        with c_query:
                            if query_img and rank == 1:
                                st.image(query_img, caption="Query", use_container_width=True)
                        with c_match:
                            img_url = f"{BACKEND_URL}/api/v1/documents/images/{hit['id']}"
                            st.image(img_url, caption=f"Match #{rank}", use_container_width=True)
                        with c_info:
                            st.subheader(f"#{rank} — {hit['file_name']}")
                            st.metric("Similarity", f"{hit['similarity']:.4f}")
                            st.write(f"**Country:** {hit['country']}  |  **Type:** {hit['doc_type']}")
                            if hit.get("ocr_text_preview"):
                                st.caption(f"OCR preview: {hit['ocr_text_preview']}...")
                    st.divider()
