"""
Stamp & Signature Verification
-------------------------------
Upload a COO or PACD document and verify the stamps/signatures present in it
against the library of known visual templates.
"""
from __future__ import annotations

import base64
import os
import sys

import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

_VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "TRUE_POSITIVE":     ("✅ TRUE POSITIVE",     "success"),
    "FALSE_POSITIVE":    ("⚠️ FALSE POSITIVE",    "warning"),
    "NO_MATCH":          ("❌ NO MATCH",           "error"),
    "HOMOGRAPHY_FAILED": ("❌ HOMOGRAPHY FAILED",  "error"),
    "NOT_RUN":           ("⏭ PHASE 2 NOT RUN",    "info"),
}

TEMPLATE_ICONS = {
    "signature": "✍️",
    "sign":      "🖊️",
    "stamp":     "📮",
    "logo":      "🏷️",
}

# ─────────────────────────────────────────────────────────────────────────────
st.header("🔏 Stamp & Signature Verification")
st.caption(
    "Upload a COO or PACD document and the system will locate and verify "
    "every registered stamp and signature within it."
)

# ── Sidebar — available templates preview ────────────────────────────────────
with st.sidebar:
    st.subheader("Registered Templates")
    _doc_cat_preview = st.selectbox(
        "Preview by category",
        ["all", "coo", "pacd"],
        key="sv_sidebar_cat",
    )
    if st.button("Refresh", key="sv_sidebar_refresh") or "sv_all_templates" not in st.session_state:
        _params: dict[str, str] = {}
        if _doc_cat_preview != "all":
            _params["doc_category"] = _doc_cat_preview
        _r = requests.get(f"{BACKEND_URL}/api/v1/visual-templates", params=_params, timeout=30)
        st.session_state["sv_all_templates"] = _r.json() if _r.ok else []

    _all_tpls = st.session_state.get("sv_all_templates", [])
    _vis_tpls = [t for t in _all_tpls if t.get("template_type") in {"signature", "sign", "stamp", "logo"}]
    if _vis_tpls:
        st.caption(f"{len(_vis_tpls)} stamp/signature template(s)")
        for _t in _vis_tpls:
            _icon = TEMPLATE_ICONS.get(_t.get("template_type", ""), "❓")
            with st.container(border=True):
                _img_r = requests.get(
                    f"{BACKEND_URL}/api/v1/visual-templates/{_t['id']}/image", timeout=15
                )
                if _img_r.ok and _img_r.content:
                    st.image(_img_r.content, use_container_width=True)
                st.caption(f"{_icon} **{_t['name']}**  \n`{_t['template_type']}` · {_t.get('doc_type') or 'any'}")
    else:
        st.info("No stamp/signature templates found. Upload some via the Visual Templates page.")

# ── Main form ─────────────────────────────────────────────────────────────────
with st.form("stamp_verify_form"):
    st.subheader("Document to Verify")

    doc_category = st.radio(
        "Document Type",
        options=["coo", "pacd"],
        format_func=lambda x: "COO — Certificate of Origin" if x == "coo" else "PACD — Pre-Arrival Customs Declaration",
        horizontal=True,
        key="sv_doc_category",
    )

    doc_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        key="sv_doc_file",
    )

    st.subheader("Filters (optional)")
    fc1, fc2 = st.columns(2)
    with fc1:
        filter_country = st.text_input(
            "Country",
            placeholder="e.g. Egypt",
            key="sv_country",
            help="Restrict to templates registered for this country",
        )
    with fc2:
        filter_doc_type = st.text_input(
            "Document sub-type",
            placeholder="e.g. AfCFTA",
            key="sv_doc_type",
            help="Restrict to templates for a specific document sub-type",
        )

    with st.expander("Advanced matching parameters"):
        threshold = st.slider(
            "Phase 1 similarity threshold",
            min_value=0.0, max_value=1.0, value=0.15, step=0.01,
            key="sv_threshold",
            help="Lower = more permissive; raise if you get too many false positives",
        )
        p2_threshold = st.slider(
            "Minimum Phase 2 inliers (LightGlue)",
            min_value=1, max_value=50, value=8, step=1,
            key="sv_p2_threshold",
            help="Number of geometric inliers required to confirm a match as TRUE POSITIVE",
        )

    submitted = st.form_submit_button(
        "▶  Verify Stamps & Signatures",
        type="primary",
        use_container_width=True,
    )

# ── Run verification ──────────────────────────────────────────────────────────
if submitted:
    if not doc_file:
        st.error("Please upload a document to verify.")
        st.stop()

    payload_data = {
        "doc_category": doc_category,
        "threshold": str(threshold),
        "p2_threshold": str(p2_threshold),
        "visual_only": "true",   # never match document_template pages
    }
    if filter_country.strip():
        payload_data["country"] = filter_country.strip()
    if filter_doc_type.strip():
        payload_data["doc_type"] = filter_doc_type.strip()

    with st.spinner("Running stamp & signature verification…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/visual-templates/match",
                data=payload_data,
                files={"file": (doc_file.name, doc_file.getvalue(), doc_file.type or "application/octet-stream")},
                timeout=300,
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if not resp.ok:
        st.error(f"Verification failed ({resp.status_code}): {resp.text}")
        st.stop()

    st.session_state["sv_result"] = resp.json()
    st.session_state["sv_doc_category_used"] = doc_category

# ── Display results ───────────────────────────────────────────────────────────
result = st.session_state.get("sv_result")
if not result:
    st.stop()

doc_cat_used = st.session_state.get("sv_doc_category_used", "document").upper()
results = result.get("results", [])
found_items = [r for r in results if r.get("found")]
total = int(result.get("count", 0))

# ── Summary banner ────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"Verification Results — {doc_cat_used}")

if result.get("any_found"):
    st.success(f"✅ Found **{len(found_items)} out of {total}** registered stamp/signature template(s) in the document.")
else:
    st.error(f"❌ None of the {total} registered stamp/signature template(s) were found in the document.")

# Show document preview
query_b64 = result.get("query_preview_jpeg_base64")
if query_b64:
    with st.expander("Document preview", expanded=False):
        st.image(base64.b64decode(query_b64), use_container_width=True)

# ── Per-template result cards ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Per-template results")

# Show found ones first, then not-found
for item in sorted(results, key=lambda x: (not x.get("found"), -float(x.get("score", 0)))):
    verdict = item.get("p2_verdict", "NOT_RUN")
    verdict_label, verdict_style = _VERDICT_STYLE.get(verdict, (verdict, "info"))
    icon = TEMPLATE_ICONS.get(item.get("template_type", ""), "❓")

    with st.container(border=True):
        col_img, col_details = st.columns([1, 3])

        with col_img:
            _img_r = requests.get(
                f"{BACKEND_URL}/api/v1/visual-templates/{item['template_id']}/image",
                timeout=15,
            )
            if _img_r.ok and _img_r.content:
                st.image(_img_r.content, use_container_width=True)
            else:
                st.caption("Image unavailable")

        with col_details:
            st.markdown(f"**{item['name']}** {icon} `{item['template_type']}`")
            getattr(st, verdict_style)(verdict_label)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("P1 Score", f"{float(item.get('score', 0)):.4f}")
            with m2:
                st.metric("LG Matches", item.get("p2_n_matches", 0))
            with m3:
                st.metric("Inliers", item.get("p2_n_inliers", 0))
            with m4:
                status_label = "Found ✅" if item.get("found") else "Not Found ❌"
                st.markdown(f"**Status**\n\n{status_label}")

            if item.get("found"):
                bbox = item.get("bounding_box", [0, 0, 0, 0])
                st.caption(f"Location (x0,y0,x1,y1): {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}")

        # Phase 2 visualisations
        hom_b64 = item.get("p2_homography_vis_jpeg_b64")
        ref_b64 = item.get("p2_refined_jpeg_b64")
        if hom_b64 or ref_b64:
            with st.expander("Phase 2 — Geometric verification", expanded=item.get("found", False)):
                v1, v2 = st.columns(2)
                if hom_b64:
                    with v1:
                        st.caption("Homography quad · inlier keypoints · bounding box")
                        st.image(base64.b64decode(hom_b64), use_container_width=True)
                if ref_b64:
                    with v2:
                        st.caption("Homography-refined region")
                        st.image(base64.b64decode(ref_b64), use_container_width=True)
