from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Step 1: Upload Document")
st.caption(
    "Upload document first. Optionally upload stamp/signature/sign/logo templates in the same step."
)

with st.form("upload_form"):
    uploaded = st.file_uploader(
        "Document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
    )
    c1, c2 = st.columns(2)
    with c1:
        country = st.text_input("Country", placeholder="e.g. Egypt")
    with c2:
        doc_type = st.text_input("Document Type", placeholder="e.g. CoO")

    st.markdown("Optional: upload template images")
    template_type = st.selectbox(
        "Template type",
        ["signature", "sign", "stamp", "logo"],
    )
    visual_templates = st.file_uploader(
        "Upload visual templates (multi-select)",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Upload and Index", type="primary", use_container_width=True)

if submitted:
    if not uploaded:
        st.error("Please select a document.")
    elif not country.strip():
        st.error("Country is required.")
    elif not doc_type.strip():
        st.error("Document Type is required.")
    else:
        with st.spinner("Preprocessing and indexing document..."):
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/documents/upload",
                data={"country": country.strip(), "doc_type": doc_type.strip()},
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=300,
            )

        if not resp.ok:
            st.error(f"Upload failed ({resp.status_code}): {resp.text}")
        else:
            data = resp.json()
            st.session_state["last_upload_country"] = country.strip()
            st.session_state["last_upload_doc_type"] = doc_type.strip()

            st.success("Document indexed successfully")
            st.markdown(f"**Pages indexed:** {data['pages_stored']}")
            st.code(data["id"], language=None)
            st.caption(f"{data['country']} | {data['doc_type']}")

            uploaded_template_ids: list[str] = []
            failed_template_uploads = 0
            if visual_templates:
                with st.spinner("Uploading visual templates..."):
                    for template in visual_templates:
                        template_resp = requests.post(
                            f"{BACKEND_URL}/api/v1/visual-templates/upload",
                            data={
                                "name": Path(template.name).stem,
                                "template_type": template_type,
                                "country": country.strip(),
                                "doc_type": doc_type.strip(),
                            },
                            files={"file": (template.name, template.getvalue(), template.type)},
                            timeout=120,
                        )
                        if template_resp.ok:
                            uploaded_template_ids.append(template_resp.json()["id"])
                        else:
                            failed_template_uploads += 1
                            st.warning(
                                f"Template '{template.name}' failed ({template_resp.status_code})"
                            )

            if uploaded_template_ids:
                st.info(f"Stored {len(uploaded_template_ids)} visual template(s)")
            if visual_templates and failed_template_uploads:
                st.warning(f"{failed_template_uploads} template file(s) could not be stored.")

            st.info("Next step: open Retrieve Template page for search and then visual detection.")
