from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import get_report, get_template_image_url, upload_coo

st.header("✅ COO Verification")
st.caption(
    "Upload a Certificate of Origin (COO) document. The system will confirm the template, "
    "extract all fields, and cross-reference them against the PACD knowledge base."
)

_VERDICT_STYLE = {
    "PASS":          ("✅ PASS — No discrepancies found",         "success"),
    "FAIL":          ("❌ FAIL — Discrepancies detected",          "error"),
    "INCONCLUSIVE":  ("⚠️ INCONCLUSIVE — Insufficient PACD data", "warning"),
}

_VERDICT_COLOR = {
    "match":              "🟢",
    "mismatch":           "🔴",
    "not_found_in_pacd":  "⚠️",
    "not_in_coo":         "🔵",
}

tab_verify, tab_report = st.tabs(["Upload & Verify", "View Report"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload & Verify
# ─────────────────────────────────────────────────────────────────────────────
with tab_verify:
    c1, c2 = st.columns(2)
    with c1:
        user_id = st.text_input(
            "User ID",
            value=st.session_state.get("user_id", ""),
            key="coo_user_id",
        )
    with c2:
        transaction_id = st.text_input(
            "Transaction ID",
            value=st.session_state.get("active_tx_id", ""),
            key="coo_tx_id",
        )

    with st.expander("Optional hints"):
        oc1, oc2 = st.columns(2)
        with oc1:
            country = st.text_input("Country hint", placeholder="e.g. EG", key="coo_country")
        with oc2:
            doc_type = st.text_input("Document type hint", placeholder="e.g. coo", key="coo_doc_type")

    coo_file = st.file_uploader(
        "COO Document",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        key="coo_file",
    )

    if st.button("Verify COO", type="primary", use_container_width=True):
        uid = user_id.strip()
        txid = transaction_id.strip()

        if not uid:
            st.error("User ID is required.")
        elif not txid:
            st.error("Transaction ID is required.")
        elif not coo_file:
            st.error("Please select a COO document.")
        else:
            steps = [
                "Ingesting document…",
                "Retrieving candidate templates…",
                "Confirming template match…",
                "Extracting COO fields…",
                "Cross-referencing against PACD…",
                "Generating verification report…",
            ]
            with st.status("Running COO verification pipeline…", expanded=True) as status_box:
                for step in steps:
                    st.write(step)

                try:
                    result = upload_coo(
                        transaction_id=txid,
                        file_bytes=coo_file.getvalue(),
                        filename=coo_file.name,
                        content_type=coo_file.type or "application/octet-stream",
                        user_id=uid,
                        country=country.strip() if country.strip() else None,
                        doc_type=doc_type.strip() if doc_type.strip() else None,
                    )
                    status_box.update(label="Verification complete!", state="complete", expanded=False)
                    st.session_state["coo_report"] = result.get("report") or {}
                    st.session_state["active_tx_id"] = txid
                    st.session_state["user_id"] = uid
                    st.rerun()  # switch to report tab
                except Exception as exc:
                    status_box.update(label="Verification failed", state="error", expanded=True)
                    st.error(f"Error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — View Report
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    # Load from session or fetch manually
    report = st.session_state.get("coo_report")

    col1, col2 = st.columns([3, 1])
    with col1:
        fetch_tx_id = st.text_input(
            "Transaction ID",
            value=st.session_state.get("active_tx_id", ""),
            key="report_tx_id",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Load Report", use_container_width=True):
            try:
                with st.spinner("Loading report…"):
                    report = get_report(fetch_tx_id.strip())
                st.session_state["coo_report"] = report
            except Exception as exc:
                st.error(f"Could not load report: {exc}")

    if report:
        # ── Verdict banner ──────────────────────────────────────────────────
        verdict = report.get("overall_verdict", "INCONCLUSIVE")
        label, style = _VERDICT_STYLE.get(verdict, (verdict, "info"))
        getattr(st, style)(label)

        # ── Template match ──────────────────────────────────────────────────
        confirmed_id = report.get("confirmed_template_id")
        confirmed_name = report.get("confirmed_template_name")
        if confirmed_id:
            with st.container(border=True):
                st.markdown("**Confirmed Template**")
                ic1, ic2 = st.columns([1, 4])
                with ic1:
                    st.image(get_template_image_url(confirmed_id), use_container_width=True)
                with ic2:
                    st.markdown(f"**{confirmed_name or confirmed_id}**")
                    st.caption(f"Template ID: {confirmed_id}")
        else:
            st.warning("No template was confirmed for this COO document.")

        st.divider()

        # ── Discrepancy Table ───────────────────────────────────────────────
        discrepancy_table = report.get("discrepancy_table") or []
        if discrepancy_table:
            st.subheader("Cross-Reference Discrepancy Table")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Fields", report.get("total_fields", len(discrepancy_table)))
            m2.metric("✅ Matched", report.get("matched", 0))
            m3.metric("❌ Mismatched", report.get("mismatched", 0))
            m4.metric("⚠️ Not Found", report.get("not_found", 0))

            table_rows = []
            for row in discrepancy_table:
                icon = _VERDICT_COLOR.get(row.get("verdict", ""), "❓")
                table_rows.append({
                    "Field": row.get("field_key", ""),
                    "COO Value": row.get("coo_value", ""),
                    "PACD Value": row.get("pacd_value") or "—",
                    "Verdict": f"{icon} {row.get('verdict', '').replace('_', ' ')}",
                    "Source Doc": row.get("pacd_source_doc") or "—",
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No cross-reference data available.")

        st.divider()

        # ── Narrative ───────────────────────────────────────────────────────
        narrative = report.get("narrative") or ""
        summary = report.get("summary") or ""
        if summary:
            st.subheader("Summary")
            st.info(summary)
        if narrative:
            st.subheader("Detailed Narrative")
            st.markdown(narrative)

        with st.expander("Raw Report JSON"):
            st.json(report)
    else:
        st.info("No report loaded. Run a COO verification or enter a Transaction ID above.")
