from __future__ import annotations

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

    for item in payload.get("results", []):
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(
                    f"{BACKEND_URL}/api/v1/visual-templates/{item['template_id']}/image",
                    use_container_width=True,
                )
            with c2:
                icon = TEMPLATE_ICONS.get(item.get("template_type", ""), "❓")
                st.markdown(f"**{item['name']}** {icon} ({item['template_type']})")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Score", f"{item['score']:.4f}")
                with m2:
                    st.markdown("**Status:** " + ("Found" if item["found"] else "Not Found"))
                if item["found"]:
                    bbox = item["bounding_box"]
                    st.caption(
                        f"Bounding box (x0,y0,x1,y1): {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}"
                    )


st.header("Visual Templates")
st.caption("Upload, manage, and test sign/signature/stamp/logo visual templates.")

tab_library, tab_upload, tab_match = st.tabs(["Library", "Upload", "Match Test"])

with tab_library:
    fc1, fc2 = st.columns(2)
    with fc1:
        filter_country = st.text_input("Country filter", placeholder="Any", key="vt_filter_country")
    with fc2:
        filter_doc_type = st.text_input("Document Type filter", placeholder="Any", key="vt_filter_doc_type")

    if st.button("Refresh Templates", key="vt_refresh_btn") or "vt_items" not in st.session_state:
        params: dict[str, str] = {}
        if filter_country.strip():
            params["country"] = filter_country.strip()
        if filter_doc_type.strip():
            params["doc_type"] = filter_doc_type.strip()

        resp = requests.get(f"{BACKEND_URL}/api/v1/visual-templates", params=params, timeout=60)
        if resp.ok:
            st.session_state["vt_items"] = resp.json()
        else:
            st.session_state["vt_items"] = []
            st.error(f"Failed to load templates ({resp.status_code}): {resp.text}")

    items: list[dict] = st.session_state.get("vt_items", [])
    if not items:
        st.info("No visual templates found for current filters.")
    else:
        st.caption(f"{len(items)} template(s)")
        for item in items:
            with st.container(border=True):
                col_img, col_meta, col_actions = st.columns([1, 2, 1])
                with col_img:
                    st.image(
                        f"{BACKEND_URL}/api/v1/visual-templates/{item['id']}/image",
                        use_container_width=True,
                    )
                with col_meta:
                    icon = TEMPLATE_ICONS.get(item.get("template_type", ""), "❓")
                    st.markdown(f"**{item['name']}** {icon}")
                    st.write(f"Type: {item['template_type']}")
                    st.write(f"Country: {item.get('country') or '-'}")
                    st.write(f"Doc Type: {item.get('doc_type') or '-'}")
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
                st.success("Template uploaded successfully")
                st.session_state.pop("vt_items", None)
            else:
                st.error(f"Upload failed ({resp.status_code}): {resp.text}")

with tab_match:
    with st.form("visual_template_match"):
        query_file = st.file_uploader(
            "Query document",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
            key="vt_query_file",
        )
        match_country = st.text_input("Country filter (optional)", key="vt_match_country")
        match_doc_type = st.text_input("Document Type filter (optional)", key="vt_match_doc_type")
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        run_match = st.form_submit_button("Run Match", type="primary", use_container_width=True)

    if run_match:
        if not query_file:
            st.error("Please upload a query document.")
        else:
            data = {
                "country": match_country.strip(),
                "doc_type": match_doc_type.strip(),
                "threshold": str(threshold),
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
