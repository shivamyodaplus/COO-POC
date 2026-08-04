from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredImage:
    data: bytes
    content_type: str


@dataclass
class VisualTemplateRecord:
    id: str
    name: str
    template_type: str
    country: str | None
    doc_type: str | None
    created_at: str


class StorageAdapter(ABC):
    @abstractmethod
    def save_document_image(
        self,
        image_id: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_document_image(self, image_id: str) -> StoredImage | None:
        raise NotImplementedError

    @abstractmethod
    def save_visual_template(
        self,
        name: str,
        template_type: str,
        country: str | None,
        doc_type: str | None,
        image_bytes: bytes,
        content_type: str,
    ) -> VisualTemplateRecord:
        raise NotImplementedError

    @abstractmethod
    def update_visual_template_attributes(
        self,
        template_id: str,
        extracted_attributes: dict,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_visual_templates(
        self,
        country: str | None,
        doc_type: str | None,
    ) -> list[VisualTemplateRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_visual_template(self, template_id: str) -> VisualTemplateRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_visual_template_image(self, template_id: str) -> StoredImage | None:
        raise NotImplementedError

    @abstractmethod
    def delete_visual_template(self, template_id: str) -> bool:
        raise NotImplementedError
