from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import get_template_status, list_visual_templates

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Step 1: Upload Template")
st.caption(
    "Upload a blank structured COO template document. "
    "Optionally upload stamps and signatures as separate visual attributes."
)

tab_template, tab_visual = st.tabs(["Blank Template Document", "Stamps & Signatures"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab A — Blank Document Template
# ──────────────────────────────────────────────────────────────────────────────
with tab_template:
    st.subheader("Upload Blank Template Document")
    st.caption(
        "Upload the blank (unfilled) version of the COO form. "
        "Supports PDF (all pages indexed) and images (PNG, JPG, TIFF, etc.)."
    )

    with st.form("template_upload_form"):
        template_file = st.file_uploader(
            "Template File",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        )
        tc1, tc2 = st.columns(2)
        with tc1:
            t_country = st.text_input("Country", placeholder="e.g. Egypt", key="t_country")
        with tc2:
            t_doc_type = st.text_input("Document Type", placeholder="e.g. CoO", key="t_doc_type")

        submitted_template = st.form_submit_button(
            "Upload & Index Template", type="primary", use_container_width=True
        )

    if submitted_template:
        if not template_file:
            st.error("Please select a template file.")
        elif not t_country.strip():
            st.error("Country is required.")
        elif not t_doc_type.strip():
            st.error("Document Type is required.")
        else:
            with st.spinner("Uploading and indexing template pages…"):
                resp = requests.post(
                    f"{BACKEND_URL}/api/v1/visual-templates/upload",
                    data={
                        "name": Path(template_file.name).stem,
                        "template_type": "document_template",
                        "country": t_country.strip(),
                        "doc_type": t_doc_type.strip(),
                        "doc_category": "coo",
                    },
                    files={
                        "file": (
                            template_file.name,
                            template_file.getvalue(),
                            template_file.type or "application/octet-stream",
                        )
                    },
                    timeout=300,
                )

            if not resp.ok:
                st.error(f"Upload failed ({resp.status_code}): {resp.text}")
            else:
                records = resp.json()
                if isinstance(records, dict):
                    records = [records]  # single page response

                st.success(f"Template indexed — {len(records)} page(s) stored.")
                st.session_state["last_template_records"] = records

                # ── Poll attribute extraction status per page ────────────────
                pending_ids = [r["id"] for r in records]
                status_placeholder = st.empty()

                poll_limit = 120  # max 120 × 3 s = 61 min
                for _ in range(poll_limit):
                    if not pending_ids:
                        break
                    time.sleep(3)
                    still_pending = []
                    for tid in pending_ids:
                        try:
                            status_data = get_template_status(tid)
                            if status_data.get("attributes_status") == "pending":
                                still_pending.append(tid)
                        except Exception:
                            still_pending.append(tid)
                    pending_ids = still_pending
                    status_placeholder.info(
                        f"Extracting field schema… {len(pending_ids)} page(s) still processing."
                        if pending_ids
                        else "Field schema extraction complete."
                    )
                    if not pending_ids:
                        break

                if pending_ids:
                    status_placeholder.warning(
                        "Field schema extraction is still running in the background. "
                        "You can proceed — it will finish shortly."
                    )
                else:
                    status_placeholder.success("All pages ready. Template is fully indexed.")

                # ── Show indexed pages ───────────────────────────────────────
                st.markdown("**Indexed Pages:**")
                for rec in records:
                    st.markdown(
                        f"- `{rec['id']}` — page {rec.get('page_num', 0)} "
                        f"| status: `{rec.get('attributes_status', 'pending')}`"
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab B — Stamps & Signatures (visual attributes, unchanged flow)
# ──────────────────────────────────────────────────────────────────────────────
with tab_visual:
    st.subheader("Upload Stamps & Signatures")
    st.caption("Upload stamp, signature, or logo images as reusable visual attributes.")

    with st.form("visual_upload_form"):
        visual_files = st.file_uploader(
            "Visual Template Images (multi-select)",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
            accept_multiple_files=True,
        )
        vc1, vc2 = st.columns(2)
        with vc1:
            v_template_type = st.selectbox(
                "Template Type", ["signature", "sign", "stamp", "logo"]
            )
        with vc2:
            v_country = st.text_input("Country (optional)", placeholder="e.g. Egypt", key="v_country")
        v_doc_type = st.text_input(
            "Document Type (optional)", placeholder="e.g. CoO", key="v_doc_type"
        )
        submitted_visual = st.form_submit_button(
            "Upload Visual Templates", type="secondary", use_container_width=True
        )

    if submitted_visual:
        if not visual_files:
            st.error("Please select at least one image.")
        else:
            success_count = 0
            fail_count = 0
            with st.spinner("Uploading visual templates…"):
                for vf in visual_files:
                    vresp = requests.post(
                        f"{BACKEND_URL}/api/v1/visual-templates/upload",
                        data={
                            "name": Path(vf.name).stem,
                            "template_type": v_template_type,
                            "country": v_country.strip() if v_country.strip() else "",
                            "doc_type": v_doc_type.strip() if v_doc_type.strip() else "",
                        },
                        files={"file": (vf.name, vf.getvalue(), vf.type or "image/png")},
                        timeout=120,
                    )
                    if vresp.ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        st.warning(f"'{vf.name}' failed ({vresp.status_code}): {vresp.text}")

            if success_count:
                st.success(f"Uploaded {success_count} visual template(s).")
            if fail_count:
                st.error(f"{fail_count} upload(s) failed.")


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
