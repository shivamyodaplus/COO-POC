from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

TEMPLATE_ICONS = {
    "signature": "✍️",
    "sign": "🖊️",
    "stamp": "📮",
    "logo": "🏷️",
}
BOX_COLORS = ["#ef4444", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6"]


def _decode_query_preview(payload: dict) -> Image.Image | None:
    b64 = payload.get("query_preview_jpeg_base64")
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _draw_match_boxes(preview: Image.Image, payload: dict) -> Image.Image:
    annotated = preview.copy()
    draw = ImageDraw.Draw(annotated)

    found_results = [item for item in payload.get("results", []) if item.get("found")]
    for idx, item in enumerate(found_results):
        bbox = item.get("bounding_box") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue

        x0, y0, x1, y1 = [int(v) for v in bbox]
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 4, max(0, y0 - 16)), f"{item['name']} ({item['score']:.2f})", fill=color)

    return annotated


def render_match_results(payload: dict) -> None:
    found_count = sum(1 for item in payload.get("results", []) if item["found"])
    total = int(payload.get("count", 0))

    if payload.get("any_found"):
        st.success(f"Found visual matches in {found_count}/{total} templates")
    else:
        st.warning(f"No visual templates matched ({total} checked)")

    preview = _decode_query_preview(payload)
    if preview is not None:
        annotated = _draw_match_boxes(preview, payload)
        st.image(annotated, caption="Query document with detected regions", use_container_width=True)

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


st.header("Step 2: Retrieve Template")
st.caption("Upload a query document and retrieve the closest indexed templates.")

with st.form("retrieve_form"):
    uploaded = st.file_uploader(
        "Query document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
    )
    col1, col2 = st.columns(2)
    with col1:
        country = st.text_input("Country filter (optional)", placeholder="e.g. Egypt")
    with col2:
        doc_type = st.text_input("Document Type filter (optional)", placeholder="e.g. CoO")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

if submitted:
    if not uploaded:
        st.error("Please upload a query document.")
    else:
        with st.spinner("Running hybrid search (dense + sparse)..."):
            form_data = {"top_k": str(top_k)}
            if country.strip():
                form_data["country"] = country.strip()
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
            st.session_state["retrieve_results"] = resp.json()
            st.session_state["retrieve_query_bytes"] = uploaded.getvalue()
            st.session_state["retrieve_query_name"] = uploaded.name
            st.session_state["retrieve_query_content_type"] = uploaded.type
            st.session_state["retrieve_country"] = country.strip()
            st.session_state["retrieve_doc_type"] = doc_type.strip()
            st.session_state.pop("retrieve_visual_match_payload", None)
            st.session_state.pop("retrieve_visual_templates", None)

results = st.session_state.get("retrieve_results")
if results is not None:
    if not results:
        st.warning("No matching templates found.")
    else:
        st.success(f"Found {len(results)} result(s)")
        st.divider()

        query_img = None
        query_bytes = st.session_state.get("retrieve_query_bytes")
        if query_bytes:
            try:
                query_img = Image.open(io.BytesIO(query_bytes))
            except Exception:
                query_img = None

        for rank, hit in enumerate(results, start=1):
            with st.container(border=True):
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

    st.subheader("Step 3: Stamp / Signature Detection")
    st.caption(
        "After retrieval, choose templates from library and/or upload new templates, then detect and visualize bounding boxes."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        sig_country = st.text_input(
            "Country filter",
            value=st.session_state.get("retrieve_country", ""),
            key="retrieve_visual_country",
        )
    with sc2:
        sig_doc_type = st.text_input(
            "Document Type filter",
            value=st.session_state.get("retrieve_doc_type", ""),
            key="retrieve_visual_doc_type",
        )

    sig_threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        key="retrieve_visual_threshold",
    )

    st.markdown("**Option A: Select Existing Templates**")
    if st.button("Load Visual Templates", key="retrieve_load_visual_templates"):
        params: dict[str, str] = {}
        if sig_country.strip():
            params["country"] = sig_country.strip()
        if sig_doc_type.strip():
            params["doc_type"] = sig_doc_type.strip()
        lresp = requests.get(
            f"{BACKEND_URL}/api/v1/visual-templates",
            params=params,
            timeout=60,
        )
        if lresp.ok:
            st.session_state["retrieve_visual_templates"] = lresp.json()
        else:
            st.error(f"Failed ({lresp.status_code}): {lresp.text}")

    retrieve_visual_templates: list[dict] = st.session_state.get("retrieve_visual_templates", [])
    options: dict[str, str] = {}
    selected_names: list[str] = []
    if retrieve_visual_templates:
        options = {
            f"{item['name']} ({item['template_type']})": item["id"]
            for item in retrieve_visual_templates
        }
        selected_names = st.multiselect(
            "Select templates",
            list(options.keys()),
            key="retrieve_visual_multiselect",
        )
    elif "retrieve_visual_templates" in st.session_state:
        st.info("No visual templates found for these filters.")

    st.markdown("**Option B: Upload Template(s) Now**")
    upload_type = st.selectbox(
        "Uploaded template type",
        ["signature", "sign", "stamp", "logo"],
        key="retrieve_visual_upload_type",
    )
    ad_hoc_templates = st.file_uploader(
        "Upload templates for this detection",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"],
        accept_multiple_files=True,
        key="retrieve_visual_upload_files",
    )

    if st.button("Run Detection", key="retrieve_run_visual_match", type="primary"):
        query_name = st.session_state.get("retrieve_query_name")
        query_content = st.session_state.get("retrieve_query_bytes")
        query_ct = st.session_state.get("retrieve_query_content_type", "application/octet-stream")

        if not query_name or not query_content:
            st.error("Query document not available. Run retrieval again.")
        else:
            template_ids: list[str] = [options[name] for name in selected_names if name in options]

            if ad_hoc_templates:
                with st.spinner("Uploading detection templates..."):
                    for template in ad_hoc_templates:
                        upload_resp = requests.post(
                            f"{BACKEND_URL}/api/v1/visual-templates/upload",
                            data={
                                "name": Path(template.name).stem,
                                "template_type": upload_type,
                                "country": sig_country.strip(),
                                "doc_type": sig_doc_type.strip(),
                            },
                            files={"file": (template.name, template.getvalue(), template.type)},
                            timeout=120,
                        )
                        if upload_resp.ok:
                            template_ids.append(upload_resp.json()["id"])
                        else:
                            st.warning(
                                f"Template '{template.name}' upload failed ({upload_resp.status_code})"
                            )

            if not template_ids:
                st.error("Select template(s) or upload template(s) before running detection.")
            else:
                mdata: list[tuple[str, str]] = [("threshold", str(sig_threshold))]
                for template_id in template_ids:
                    mdata.append(("template_ids", template_id))

                with st.spinner("Detecting visual regions..."):
                    mresp = requests.post(
                        f"{BACKEND_URL}/api/v1/visual-templates/match",
                        data=mdata,
                        files={"file": (query_name, query_content, query_ct)},
                        timeout=300,
                    )

                if mresp.ok:
                    st.session_state["retrieve_visual_match_payload"] = mresp.json()
                else:
                    st.error(f"Match failed ({mresp.status_code}): {mresp.text}")

    if st.session_state.get("retrieve_visual_match_payload"):
        render_match_results(st.session_state["retrieve_visual_match_payload"])
