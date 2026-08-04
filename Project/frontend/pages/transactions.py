from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import create_transaction, list_user_transactions

st.header("📋 Transactions")
st.caption("Create a new transaction or review your transaction history.")

# ── Create New Transaction ──────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Start New Transaction")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_user_id = st.text_input(
            "User ID",
            value=st.session_state.get("user_id", ""),
            key="tx_new_user_id",
            placeholder="e.g. user123",
        )
    with col2:
        st.write("")
        st.write("")
        create_btn = st.button("Create Transaction", type="primary", use_container_width=True)

    if create_btn:
        if not new_user_id.strip():
            st.error("User ID is required.")
        else:
            try:
                with st.spinner("Creating transaction…"):
                    tx = create_transaction(new_user_id.strip())
                st.session_state["user_id"] = new_user_id.strip()
                st.session_state["active_tx_id"] = tx["id"]
                st.success(f"Transaction created!")
                st.markdown(f"**Transaction ID:**")
                st.code(tx["id"], language=None)
                st.caption(f"Status: {tx['status']} | Created: {tx['created_at']}")
            except Exception as exc:
                st.error(f"Failed to create transaction: {exc}")

if st.session_state.get("active_tx_id"):
    st.info(f"Active transaction: `{st.session_state['active_tx_id']}`")

st.divider()

# ── My Transactions ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("My Transactions")
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_user_id = st.text_input(
            "User ID to look up",
            value=st.session_state.get("user_id", ""),
            key="tx_lookup_user_id",
            placeholder="e.g. user123",
        )
    with col2:
        st.write("")
        st.write("")
        load_btn = st.button("Load Transactions", use_container_width=True)

    if load_btn:
        if not lookup_user_id.strip():
            st.error("User ID is required.")
        else:
            try:
                with st.spinner("Loading…"):
                    txns = list_user_transactions(lookup_user_id.strip())
                st.session_state["user_id"] = lookup_user_id.strip()
                st.session_state["txn_list"] = txns
            except Exception as exc:
                st.error(f"Failed to load transactions: {exc}")

    txns_to_show = st.session_state.get("txn_list", [])
    if txns_to_show:
        st.markdown(f"**{len(txns_to_show)} transaction(s) found**")
        for tx in txns_to_show:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.markdown(f"**ID:** `{tx['id']}`")
                with c2:
                    status_color = "🟢" if tx["status"] == "active" else "🔴"
                    st.markdown(f"**Status:** {status_color} {tx['status']}")
                with c3:
                    if st.button("Select", key=f"select_tx_{tx['id']}", use_container_width=True):
                        st.session_state["active_tx_id"] = tx["id"]
                        st.success(f"Active transaction set to `{tx['id']}`")
                st.caption(f"Created: {tx['created_at']}")
    elif load_btn:
        st.info("No transactions found for this user.")
