from __future__ import annotations

import base64
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

TEMPLATE_ICONS = {
    "signature": "✍️",
    "sign": "🖊️",
    "stamp": "📮",
    "logo": "🏷️",
}


def render_match_results(payload: dict) -> None:
    found_count = sum(1 for item in payload.get("results", []) if item["found"])
    total = int(payload.get("count", 0))

    if payload.get("any_found"):
        st.success(f"Found visual matches in {found_count}/{total} templates")
    else:
        st.warning(f"No visual templates matched ({total} checked)")

    _VERDICT_STYLE: dict[str, tuple[str, str]] = {
        "TRUE_POSITIVE":    ("✅ TRUE POSITIVE",    "success"),
        "FALSE_POSITIVE":   ("⚠️ FALSE POSITIVE",   "warning"),
        "NO_MATCH":         ("❌ NO MATCH",          "error"),
        "HOMOGRAPHY_FAILED":("❌ HOMOGRAPHY FAILED", "error"),
        "NOT_RUN":          ("⏭️ PHASE 2 NOT RUN",   "info"),
    }

    for item in payload.get("results", []):
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                try:
                    _vt_r = requests.get(
                        f"{BACKEND_URL}/api/v1/visual-templates/{item['template_id']}/image",
                        timeout=30,
                    )
                    st.image(_vt_r.content if _vt_r.ok else b"", use_container_width=True)
                except Exception:
                    st.caption("Image unavailable")
            with c2:
                icon = TEMPLATE_ICONS.get(item.get("template_type", ""), "❓")
                st.markdown(f"**{item['name']}** {icon} ({item['template_type']})")

                # ── Verdict badge ─────────────────────────────────────────────
                verdict = item.get("p2_verdict", "NOT_RUN")
                label, style = _VERDICT_STYLE.get(verdict, (verdict, "info"))
                getattr(st, style)(label)

                # ── Metrics row ───────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("P1 Score", f"{item['score']:.4f}")
                with m2:
                    st.metric("LG Matches", item.get("p2_n_matches", 0))
                with m3:
                    st.metric("Inliers", item.get("p2_n_inliers", 0))
                with m4:
                    st.markdown("**Status:** " + ("Found" if item["found"] else "Not Found"))

                if item["found"]:
                    bbox = item["bounding_box"]
                    st.caption(
                        f"Bounding box (x0,y0,x1,y1): {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}"
                    )

            # ── Phase 2 visualisations ─────────────────────────────────────────
            hom_b64 = item.get("p2_homography_vis_jpeg_b64")
            ref_b64 = item.get("p2_refined_jpeg_b64")
            if hom_b64 or ref_b64:
                with st.expander("Phase 2 Verification", expanded=item["found"]):
                    v1, v2 = st.columns(2)
                    if hom_b64:
                        with v1:
                            st.caption("Homography quad · inlier kps · bbox")
                            st.image(base64.b64decode(hom_b64), use_container_width=True)
                    if ref_b64:
                        with v2:
                            st.caption("Homography-refined region")
                            st.image(base64.b64decode(ref_b64), use_container_width=True)


st.header("Visual Templates")
st.caption("Signatures and stamps used to verify documents. Upload, manage, and test templates.")

tab_library, tab_upload, tab_match = st.tabs(["Library", "Upload", "Match Test"])

with tab_library:
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        filter_country = st.text_input("Country filter", placeholder="Any", key="vt_filter_country")
    with fc2:
        filter_doc_type = st.text_input("Document Type filter", placeholder="Any", key="vt_filter_doc_type")
    with fc3:
        filter_doc_category = st.selectbox(
            "Document Category", ["any", "coo", "pacd"], key="vt_filter_doc_category"
        )
    with fc4:
        filter_template_type = st.selectbox(
            "Template Type",
            ["all", "signature", "sign", "stamp", "logo", "document_template"],
            key="vt_filter_template_type",
        )

    if st.button("Refresh Templates", key="vt_refresh_btn") or "vt_items" not in st.session_state:
        params: dict[str, str] = {}
        if filter_country.strip():
            params["country"] = filter_country.strip()
        if filter_doc_type.strip():
            params["doc_type"] = filter_doc_type.strip()
        if filter_doc_category and filter_doc_category != "any":
            params["doc_category"] = filter_doc_category

        resp = requests.get(f"{BACKEND_URL}/api/v1/visual-templates", params=params, timeout=60)
        if resp.ok:
            st.session_state["vt_items"] = resp.json()
        else:
            st.session_state["vt_items"] = []
            st.error(f"Failed to load templates ({resp.status_code}): {resp.text}")

    all_items: list[dict] = st.session_state.get("vt_items", [])

    # Split into stamps/signatures and document templates
    stamp_sig_items = [i for i in all_items if i.get("template_type") in {"signature", "sign", "stamp", "logo"}]
    doc_template_items = [i for i in all_items if i.get("template_type") == "document_template"]

    if filter_template_type == "all":
        items = all_items
    elif filter_template_type == "document_template":
        items = doc_template_items
    else:
        items = [i for i in all_items if i.get("template_type") == filter_template_type]

    # Show a summary banner
    if stamp_sig_items:
        st.success(
            f"**{len(stamp_sig_items)} signature/stamp template(s)** loaded — "
            f"{sum(1 for i in stamp_sig_items if i.get('template_type') == 'stamp')} stamp(s), "
            f"{sum(1 for i in stamp_sig_items if i.get('template_type') in ('signature', 'sign'))} signature(s), "
            f"{sum(1 for i in stamp_sig_items if i.get('template_type') == 'logo')} logo(s)"
        )
    else:
        st.warning(
            "No signature or stamp templates found. "
            "Upload signature/stamp images in the **Upload** tab to enable visual verification."
        )

    if not items:
        st.info("No templates found for the current filter.")
    else:
        st.caption(f"{len(items)} template(s) shown")
        for item in items:
            is_stamp_sig = item.get("template_type") in {"signature", "sign", "stamp", "logo"}
            with st.container(border=True):
                col_img, col_meta, col_actions = st.columns([1, 2, 1])
                with col_img:
                    try:
                        _r = requests.get(
                            f"{BACKEND_URL}/api/v1/visual-templates/{item['id']}/image",
                            timeout=30,
                        )
                        if _r.ok and _r.content:
                            st.image(_r.content, use_container_width=True)
                        else:
                            st.caption("Image unavailable")
                    except Exception:
                        st.caption("Image unavailable")
                with col_meta:
                    icon = TEMPLATE_ICONS.get(item.get("template_type", ""), "❓")
                    st.markdown(f"**{item['name']}** {icon}")
                    st.write(f"Type: `{item['template_type']}`")
                    st.write(f"Country: {item.get('country') or '-'}")
                    st.write(f"Doc Type: {item.get('doc_type') or '-'}")
                    st.write(f"Category: {item.get('doc_category') or '-'}")
                    if is_stamp_sig:
                        st.caption("🔍 Used for document verification")
                    st.caption(f"ID: {item['id']}")
                with col_actions:
                    if st.button("Delete", key=f"del_{item['id']}"):
                        dresp = requests.delete(
                            f"{BACKEND_URL}/api/v1/visual-templates/{item['id']}",
                            timeout=60,
                        )
                        if dresp.ok:
                            st.session_state.pop("vt_items", None)
                            st.rerun()
                        else:
                            st.error(f"Delete failed ({dresp.status_code})")

with tab_upload:
    with st.form("visual_template_upload"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Template Name", placeholder="e.g. Director Signature")
            template_type = st.selectbox("Template Type", ["signature", "sign", "stamp", "logo"])
        with c2:
            country = st.text_input("Country (optional)", placeholder="e.g. Egypt")
            doc_type = st.text_input("Document Type (optional)", placeholder="e.g. CoO")

        template_file = st.file_uploader(
            "Template image",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        )
        submitted = st.form_submit_button("Upload Template", type="primary", use_container_width=True)

    if submitted:
        if not template_file:
            st.error("Please select a template image.")
        elif not name.strip():
            st.error("Template Name is required.")
        else:
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/visual-templates/upload",
                data={
                    "name": name.strip(),
                    "template_type": template_type,
                    "country": country.strip(),
                    "doc_type": doc_type.strip(),
                },
                files={"file": (template_file.name, template_file.getvalue(), template_file.type)},
                timeout=120,
            )
            if resp.ok:
                record = resp.json()
                st.success("Template uploaded successfully")
                st.session_state.pop("vt_items", None)
                # Show extracted attributes if already populated (may be None while background task runs)
                attrs = record.get("extracted_attributes")
                if attrs:
                    st.info(
                        "**Fields detected on this template:** "
                        + ", ".join(f"`{k}`" for k in attrs.keys())
                    )
                else:
                    st.caption("⏳ Extracting template field attributes in the background…")
            else:
                st.error(f"Upload failed ({resp.status_code}): {resp.text}")

with tab_match:
    with st.form("visual_template_match"):
        query_file = st.file_uploader(
            "Query document",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
            key="vt_query_file",
        )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            match_country = st.text_input("Country filter (optional)", key="vt_match_country")
        with mc2:
            match_doc_type = st.text_input("Document Type filter (optional)", key="vt_match_doc_type")
        with mc3:
            match_doc_category = st.selectbox(
                "Document Category", ["any", "coo", "pacd"], key="vt_match_doc_category"
            )
        threshold = st.slider("Phase 1 Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        p2_threshold = st.slider(
            "Min Phase 2 Inliers (LightGlue)",
            min_value=1, max_value=50, value=10, step=1,
            key="vt_p2_threshold",
            help="Minimum MAGSAC++ inliers needed to confirm a TRUE POSITIVE",
        )
        run_match = st.form_submit_button("Run Match", type="primary", use_container_width=True)

    if run_match:
        if not query_file:
            st.error("Please upload a query document.")
        else:
            data = {
                "country": match_country.strip(),
                "doc_type": match_doc_type.strip(),
                "doc_category": match_doc_category if match_doc_category != "any" else "",
                "threshold": str(threshold),
                "p2_threshold": str(p2_threshold),
            }
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/visual-templates/match",
                data=data,
                files={"file": (query_file.name, query_file.getvalue(), query_file.type)},
                timeout=300,
            )
            if not resp.ok:
                st.error(f"Match failed ({resp.status_code}): {resp.text}")
            else:
                st.session_state["vt_match_payload"] = resp.json()

    if st.session_state.get("vt_match_payload"):
        render_match_results(st.session_state["vt_match_payload"])
