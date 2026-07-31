from __future__ import annotations

import uuid
from typing import Any

from app.adapters.base import StorageAdapter, StoredImage, VisualTemplateRecord


class PostgresStorageAdapter(StorageAdapter):
    def __init__(self, pool: Any):
        self._pool = pool

    def save_document_image(
        self,
        image_id: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET image_data = %s,
                        image_content_type = %s
                    WHERE id = %s
                    """,
                    (image_bytes, content_type, image_id),
                )
            conn.commit()

    def get_document_image(self, image_id: str) -> StoredImage | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_data, image_content_type
                FROM documents
                WHERE id = %s
                """,
                (image_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return StoredImage(
            data=bytes(row[0]),
            content_type=row[1] or "image/jpeg",
        )

    def save_visual_template(
        self,
        name: str,
        template_type: str,
        country: str | None,
        doc_type: str | None,
        image_bytes: bytes,
        content_type: str,
    ) -> VisualTemplateRecord:
        template_id = str(uuid.uuid4())
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO visual_templates (
                    id,
                    name,
                    template_type,
                    country,
                    doc_type,
                    image_data,
                    image_content_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING created_at
                """,
                (
                    template_id,
                    name,
                    template_type,
                    country,
                    doc_type,
                    image_bytes,
                    content_type,
                ),
            )
            row = cur.fetchone()
            conn.commit()

        return VisualTemplateRecord(
            id=template_id,
            name=name,
            template_type=template_type,
            country=country,
            doc_type=doc_type,
            created_at=row[0].isoformat(),
        )

    def list_visual_templates(
        self,
        country: str | None,
        doc_type: str | None,
    ) -> list[VisualTemplateRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if country:
            clauses.append("country = %s")
            params.append(country)
        if doc_type:
            clauses.append("doc_type = %s")
            params.append(doc_type)

        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

        query = f"""
            SELECT id, name, template_type, country, doc_type, created_at
            FROM visual_templates
            {where_sql}
            ORDER BY created_at DESC
        """

        out: list[VisualTemplateRecord] = []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query, tuple(params))
            for row in cur.fetchall():
                out.append(
                    VisualTemplateRecord(
                        id=str(row[0]),
                        name=row[1],
                        template_type=row[2],
                        country=row[3],
                        doc_type=row[4],
                        created_at=row[5].isoformat(),
                    )
                )
        return out

    def get_visual_template(self, template_id: str) -> VisualTemplateRecord | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, template_type, country, doc_type, created_at
                FROM visual_templates
                WHERE id = %s
                """,
                (template_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        return VisualTemplateRecord(
            id=str(row[0]),
            name=row[1],
            template_type=row[2],
            country=row[3],
            doc_type=row[4],
            created_at=row[5].isoformat(),
        )

    def get_visual_template_image(self, template_id: str) -> StoredImage | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_data, image_content_type
                FROM visual_templates
                WHERE id = %s
                """,
                (template_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        return StoredImage(
            data=bytes(row[0]),
            content_type=row[1] or "application/octet-stream",
        )

    def delete_visual_template(self, template_id: str) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM visual_templates WHERE id = %s", (template_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
