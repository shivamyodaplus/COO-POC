from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.adapters.base import StorageAdapter, StoredImage, VisualTemplateRecord  # noqa: F401

IMAGE_DIR = Path("/app/images")
TEMPLATE_DIR = Path("/app/visual_templates")
TEMPLATE_INDEX = TEMPLATE_DIR / "index.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


class LocalStorageAdapter(StorageAdapter):
    def save_document_image(
        self,
        image_id: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> None:
        path = IMAGE_DIR / f"{image_id}.jpg"
        path.write_bytes(image_bytes)

    def get_document_image(self, image_id: str) -> StoredImage | None:
        path = IMAGE_DIR / f"{image_id}.jpg"
        if not path.exists():
            return None
        return StoredImage(data=path.read_bytes(), content_type="image/jpeg")

    def save_visual_template(
        self,
        name: str,
        template_type: str,
        country: str | None,
        doc_type: str | None,
        image_bytes: bytes,
        content_type: str,
        page_num: int = 0,
    ) -> VisualTemplateRecord:
        template_id = str(uuid.uuid4())
        image_path = TEMPLATE_DIR / f"{template_id}.bin"
        image_path.write_bytes(image_bytes)

        record = {
            "id": template_id,
            "name": name,
            "template_type": template_type,
            "country": country,
            "doc_type": doc_type,
            "content_type": content_type,
            "page_num": page_num,
            "attributes_status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
        }

        entries = self._read_entries()
        entries.append(record)
        TEMPLATE_INDEX.write_text(json.dumps(entries), encoding="utf-8")

        return VisualTemplateRecord(
            id=template_id,
            name=name,
            template_type=template_type,
            country=country,
            doc_type=doc_type,
            created_at=record["created_at"],
            page_num=page_num,
            attributes_status="pending",
        )

    def list_visual_templates(
        self,
        country: str | None,
        doc_type: str | None,
    ) -> list[VisualTemplateRecord]:
        entries = self._read_entries()
        out: list[VisualTemplateRecord] = []
        for entry in entries:
            if country and entry.get("country") != country:
                continue
            if doc_type and entry.get("doc_type") != doc_type:
                continue
            out.append(
                VisualTemplateRecord(
                    id=entry["id"],
                    name=entry["name"],
                    template_type=entry["template_type"],
                    country=entry.get("country"),
                    doc_type=entry.get("doc_type"),
                    created_at=entry["created_at"],
                )
            )
        out.sort(key=lambda x: x.created_at, reverse=True)
        return out

    def get_visual_template(self, template_id: str) -> VisualTemplateRecord | None:
        for entry in self._read_entries():
            if entry["id"] == template_id:
                return VisualTemplateRecord(
                    id=entry["id"],
                    name=entry["name"],
                    template_type=entry["template_type"],
                    country=entry.get("country"),
                    doc_type=entry.get("doc_type"),
                    created_at=entry["created_at"],
                )
        return None

    def get_visual_template_image(self, template_id: str) -> StoredImage | None:
        entries = self._read_entries()
        for entry in entries:
            if entry["id"] == template_id:
                image_path = TEMPLATE_DIR / f"{template_id}.bin"
                if not image_path.exists():
                    return None
                return StoredImage(
                    data=image_path.read_bytes(),
                    content_type=entry.get("content_type", "application/octet-stream"),
                )
        return None

    def update_visual_template_attributes(
        self,
        template_id: str,
        extracted_attributes: dict,
    ) -> None:
        entries = self._read_entries()
        for entry in entries:
            if entry["id"] == template_id:
                entry["extracted_attributes"] = extracted_attributes  # type: ignore[assignment]
                break
        TEMPLATE_INDEX.write_text(json.dumps(entries), encoding="utf-8")

    def delete_visual_template(self, template_id: str) -> bool:
        entries = self._read_entries()
        kept = [entry for entry in entries if entry.get("id") != template_id]
        if len(kept) == len(entries):
            return False

        TEMPLATE_INDEX.write_text(json.dumps(kept), encoding="utf-8")
        image_path = TEMPLATE_DIR / f"{template_id}.bin"
        image_path.unlink(missing_ok=True)
        return True

    def update_template_status(self, template_id: str, status: str) -> None:
        entries = self._read_entries()
        for entry in entries:
            if entry["id"] == template_id:
                entry["attributes_status"] = status  # type: ignore[assignment]
                break
        TEMPLATE_INDEX.write_text(json.dumps(entries), encoding="utf-8")

    def _read_entries(self) -> list[dict[str, str | None]]:
        if not TEMPLATE_INDEX.exists():
            return []
        raw = TEMPLATE_INDEX.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return data
