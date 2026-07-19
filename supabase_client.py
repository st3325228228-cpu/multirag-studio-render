"""
supabase_client.py
負責與 Supabase 的連線、寫入問答紀錄、讀取歷史紀錄。
支援兩種金鑰來源：UI 手動輸入（優先） 或 secrets.toml（備援）。
若未設定金鑰，所有函式會安全地回傳空結果，不會拋出例外中斷主程式。
"""

import streamlit as st

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None


def get_supabase_client(url: str = "", key: str = ""):
    """
    建立並回傳 Supabase client。
    優先使用傳入的 url/key（來自 UI 輸入），
    若未傳入，則退回讀取 st.secrets（來自 secrets.toml）。
    若都沒有設定，回傳 None。
    """
    if create_client is None:
        return None

    # 優先使用 UI 輸入的值
    final_url = url.strip() if url else ""
    final_key = key.strip() if key else ""

    # 沒有 UI 輸入時，退回讀取 secrets.toml
    if not final_url or not final_key:
        try:
            final_url = final_url or st.secrets["supabase"]["url"]
            final_key = final_key or st.secrets["supabase"]["key"]
        except Exception:
            pass

    if not final_url or not final_key:
        return None

    try:
        return create_client(final_url, final_key)
    except Exception:
        return None


def save_qa_record(client, question: str, answer: str, strategy: str) -> bool:
    """寫入一筆問答紀錄，成功回傳 True，失敗回傳 False。"""
    if client is None:
        return False
    try:
        client.table("qa_history").insert({
            "question": question,
            "answer": answer,
            "strategy": strategy,
        }).execute()
        return True
    except Exception:
        return False


def load_qa_history(client, limit: int = 20) -> list[dict]:
    """讀取最近的問答紀錄，若無法連線則回傳空清單。"""
    if client is None:
        return []
    try:
        result = (
            client.table("qa_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        return []
