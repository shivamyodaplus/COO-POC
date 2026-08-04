"""
Centralised HTTP client for the Streamlit frontend.

All pages import from here instead of calling requests.post(...) directly.
The BACKEND_URL is read once from the environment.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_BASE = f"{BACKEND_URL}/api/v1"

_TIMEOUT_SHORT = 30
_TIMEOUT_LONG = 600  # long for LLM inference


# --------------------------------------------------------------------------- #
# Transactions                                                                 #
# --------------------------------------------------------------------------- #

def create_transaction(user_id: str) -> dict[str, Any]:
    resp = requests.post(f"{_BASE}/transactions", json={"user_id": user_id}, timeout=_TIMEOUT_SHORT)
    resp.raise_for_status()
    return resp.json()


def get_transaction(transaction_id: str) -> dict[str, Any]:
    resp = requests.get(f"{_BASE}/transactions/{transaction_id}", timeout=_TIMEOUT_SHORT)
    resp.raise_for_status()
    return resp.json()


def list_user_transactions(user_id: str) -> list[dict[str, Any]]:
    resp = requests.get(f"{_BASE}/users/{user_id}/transactions", timeout=_TIMEOUT_SHORT)
    resp.raise_for_status()
    return resp.json()


def get_report(transaction_id: str) -> dict[str, Any]:
    resp = requests.get(f"{_BASE}/transactions/{transaction_id}/report", timeout=_TIMEOUT_SHORT)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# PACD ingestion                                                               #
# --------------------------------------------------------------------------- #

def upload_pacd(
    transaction_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: str,
) -> dict[str, Any]:
    resp = requests.post(
        f"{_BASE}/transactions/{transaction_id}/pacd",
        data={"user_id": user_id},
        files={"file": (filename, file_bytes, content_type)},
        timeout=_TIMEOUT_LONG,
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# COO verification                                                             #
# --------------------------------------------------------------------------- #

def upload_coo(
    transaction_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: str,
    country: str | None = None,
    doc_type: str | None = None,
) -> dict[str, Any]:
    data: dict[str, str] = {"user_id": user_id}
    if country:
        data["country"] = country
    if doc_type:
        data["doc_type"] = doc_type
    resp = requests.post(
        f"{_BASE}/transactions/{transaction_id}/coo",
        data=data,
        files={"file": (filename, file_bytes, content_type)},
        timeout=_TIMEOUT_LONG,
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Visual templates                                                             #
# --------------------------------------------------------------------------- #

def list_visual_templates(country: str | None = None, doc_type: str | None = None) -> list[dict]:
    params: dict[str, str] = {}
    if country:
        params["country"] = country
    if doc_type:
        params["doc_type"] = doc_type
    resp = requests.get(f"{_BASE}/visual-templates", params=params, timeout=_TIMEOUT_SHORT)
    resp.raise_for_status()
    return resp.json()


def get_template_image_url(template_id: str) -> str:
    return f"{_BASE}/visual-templates/{template_id}/image"


def get_template_status(template_id: str) -> dict[str, Any]:
    """Poll extraction status for a single template record."""
    resp = requests.get(
        f"{_BASE}/visual-templates/{template_id}/status", timeout=_TIMEOUT_SHORT
    )
    resp.raise_for_status()
    return resp.json()


def list_templates_by_type(
    template_type: str,
    country: str | None = None,
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return visual templates filtered by type (e.g. 'document_template')."""
    params: dict[str, str] = {}
    if country:
        params["country"] = country
    if doc_type:
        params["doc_type"] = doc_type
    items = list_visual_templates(country=country, doc_type=doc_type)
    return [i for i in items if i.get("template_type") == template_type]
