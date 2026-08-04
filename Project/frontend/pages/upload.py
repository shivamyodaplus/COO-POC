from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import upload_pacd

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

    # Optional: link this upload to an existing transaction (routes to PACD ingestion)
    st.markdown("---")
    st.markdown("**Link to Transaction (optional)**")
    st.caption("If filled, the document will be ingested as a PACD document for that transaction.")
    tc1, tc2 = st.columns(2)
    with tc1:
        tx_user_id = st.text_input("User ID", placeholder="e.g. user123", key="upload_tx_user_id")
    with tc2:
        tx_id = st.text_input("Transaction ID", placeholder="leave blank for standalone upload", key="upload_tx_id")

    st.markdown("---")
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
        # Route to PACD ingestion if transaction context is provided
        use_pacd_route = bool(tx_user_id.strip() and tx_id.strip())

        if use_pacd_route:
            with st.spinner("Ingesting PACD document into knowledge base…"):
                try:
                    data = upload_pacd(
                        transaction_id=tx_id.strip(),
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        content_type=uploaded.type or "application/octet-stream",
                        user_id=tx_user_id.strip(),
                    )
                    st.session_state["active_tx_id"] = tx_id.strip()
                    st.session_state["user_id"] = tx_user_id.strip()
                    st.success("PACD document ingested into knowledge base")
                    st.markdown(f"**Pages indexed:** {data.get('pages_indexed', 0)}")
                    st.code(data.get("document_id", ""), language=None)
                    kv_pairs = data.get("extracted_kv_pairs") or []
                    if kv_pairs:
                        st.info(f"{len(kv_pairs)} key-value pairs extracted")
                    if data.get("errors"):
                        for err in data["errors"]:
                            st.warning(err)
                except Exception as exc:
                    st.error(f"PACD ingestion failed: {exc}")
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

        if not use_pacd_route:
            st.info("Next step: open Retrieve Template page for search and then visual detection.")
