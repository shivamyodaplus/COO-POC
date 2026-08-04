from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import upload_pacd

st.header("📂 Upload PACD Documents")
st.caption(
    "Upload Pre-Arrival Customs Declaration (PACD) documents for a transaction. "
    "The Vision LLM will extract all key-value fields from each page and store them "
    "in the knowledge base for cross-reference during COO verification."
)

# ── Transaction context ──────────────────────────────────────────────────────
with st.expander("Transaction Context", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        user_id = st.text_input(
            "User ID",
            value=st.session_state.get("user_id", ""),
            key="pacd_user_id",
        )
    with c2:
        transaction_id = st.text_input(
            "Transaction ID",
            value=st.session_state.get("active_tx_id", ""),
            key="pacd_tx_id",
        )

    if user_id:
        st.session_state["user_id"] = user_id
    if transaction_id:
        st.session_state["active_tx_id"] = transaction_id

# ── File uploader ────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "PACD Documents (multi-select)",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    accept_multiple_files=True,
    key="pacd_files",
)

if st.button("Upload & Index PACD Documents", type="primary", use_container_width=True):
    uid = st.session_state.get("pacd_user_id", "").strip()
    txid = st.session_state.get("pacd_tx_id", "").strip()

    if not uid:
        st.error("User ID is required.")
    elif not txid:
        st.error("Transaction ID is required. Create one on the Transactions page first.")
    elif not uploaded_files:
        st.error("Please select at least one PACD document.")
    else:
        progress = st.progress(0, text="Preparing uploads…")
        results = []
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            progress.progress((i) / total, text=f"Processing {f.name} ({i + 1}/{total})…")
            try:
                result = upload_pacd(
                    transaction_id=txid,
                    file_bytes=f.getvalue(),
                    filename=f.name,
                    content_type=f.type or "application/octet-stream",
                    user_id=uid,
                )
                results.append({"filename": f.name, "ok": True, "data": result})
            except Exception as exc:
                results.append({"filename": f.name, "ok": False, "error": str(exc)})

        progress.progress(1.0, text="Done!")

        st.markdown("---")
        st.subheader("Results")

        for r in results:
            if r["ok"]:
                d = r["data"]
                with st.expander(f"✅ {r['filename']} — {d.get('pages_indexed', 0)} page(s) indexed", expanded=False):
                    kv_pairs = d.get("extracted_kv_pairs") or []
                    if kv_pairs:
                        st.markdown("**Extracted Key-Value Pairs (sample):**")
                        # Group by page for display
                        by_page: dict[int, list[dict]] = {}
                        for kv in kv_pairs:
                            by_page.setdefault(kv.get("page", 1), []).append(kv)
                        for page_num in sorted(by_page.keys()):
                            st.markdown(f"*Page {page_num}*")
                            table_data = [
                                {"Field": kv["key"], "Value": kv["value"], "Confidence": f"{kv.get('confidence', 0):.0%}"}
                                for kv in by_page[page_num]
                            ]
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("No key-value pairs were extracted from this document.")
                    if d.get("errors"):
                        for err in d["errors"]:
                            st.warning(err)
            else:
                st.error(f"❌ {r['filename']}: {r['error']}")
