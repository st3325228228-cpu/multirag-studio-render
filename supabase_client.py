"""
supabase_client.py
負責與 Supabase 的連線、寫入問答紀錄、讀取歷史紀錄。
支援三種金鑰來源，依優先順序為：
1. UI 手動輸入（最優先，適合本機或雲端臨時測試）
2. secrets.toml（本機開發環境備援）
3. 環境變數 os.environ（Render 等雲端部署環境備援）
若都未設定金鑰，所有函式會安全地回傳空結果，不會拋出例外中斷主程式。
"""

import os
import streamlit as st

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None


def get_supabase_client(url: str = "", key: str = ""):
    """
    建立並回傳 Supabase client。
    優先順序：
    1. 傳入的 url/key（來自 UI 輸入）
    2. st.secrets（來自本機 secrets.toml）
    3. os.environ（來自 Render 等雲端平台的環境變數）
    若三者都沒有設定，回傳 None。
    """
    if create_client is None:
        return None

    # 第一優先：使用 UI 輸入的值
    final_url = url.strip() if url else ""
    final_key = key.strip() if key else ""

    # 第二優先：退回讀取 secrets.toml（本機開發環境）
    if not final_url or not final_key:
        try:
            final_url = final_url or st.secrets["supabase"]["url"]
            final_key = final_key or st.secrets["supabase"]["key"]
        except Exception:
            pass

    # 第三優先：退回讀取環境變數（Render 等雲端部署環境）
    if not final_url or not final_key:
        final_url = final_url or os.environ.get("SUPABASE_URL", "")
        final_key = final_key or os.environ.get("SUPABASE_KEY", "")

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
