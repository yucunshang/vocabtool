# Optional Google Sheets-backed persistent cache.
#
# This module is intentionally best-effort:
# - If configuration is missing, cache is disabled silently.
# - If Google libs are missing, cache is disabled silently.
# - Runtime errors never break the main app flow.

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_READY = False
_ENABLED = False
_WORKSHEET = None
_CACHE_MAP: Dict[str, str] = {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_streamlit_secrets() -> Dict[str, Any]:
    try:
        import streamlit as st
        return getattr(st, "secrets", None) or {}
    except Exception:
        return {}


def _read_secret_or_env(
    secrets: Dict[str, Any],
    secret_key: str,
    env_key: Optional[str] = None,
    default: Any = "",
) -> Any:
    env_name = env_key or secret_key
    secret_val = None
    try:
        secret_val = secrets.get(secret_key)
    except Exception:
        secret_val = None
    if secret_val not in (None, ""):
        return secret_val
    return os.environ.get(env_name, default)


def _parse_service_account(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            logger.warning("GOOGLE service account JSON is invalid.")
            return None
    return None


def _load_settings() -> Tuple[bool, str, str, Optional[Dict[str, Any]]]:
    secrets = _get_streamlit_secrets()

    enabled = _as_bool(
        _read_secret_or_env(
            secrets,
            "GOOGLE_SHEETS_CACHE_ENABLED",
            default="0",
        )
    )

    spreadsheet_id = _read_secret_or_env(
        secrets,
        "GOOGLE_SHEETS_CACHE_SPREADSHEET_ID",
        default="",
    )
    if not spreadsheet_id:
        spreadsheet_id = _read_secret_or_env(
            secrets,
            "GOOGLE_SHEETS_SPREADSHEET_ID",
            default="",
        )

    worksheet_name = _read_secret_or_env(
        secrets,
        "GOOGLE_SHEETS_CACHE_WORKSHEET",
        default="cache",
    )

    raw_sa = (
        _read_secret_or_env(secrets, "GOOGLE_SHEETS_SERVICE_ACCOUNT", default=None)
        or _read_secret_or_env(secrets, "GOOGLE_SERVICE_ACCOUNT", default=None)
        or _read_secret_or_env(secrets, "GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON", default=None)
        or _read_secret_or_env(secrets, "GOOGLE_SERVICE_ACCOUNT_JSON", default=None)
    )
    service_account = _parse_service_account(raw_sa)
    return enabled, str(spreadsheet_id).strip(), str(worksheet_name).strip() or "cache", service_account


def _ensure_ready() -> None:
    global _READY, _ENABLED, _WORKSHEET
    with _LOCK:
        if _READY:
            return

        enabled, spreadsheet_id, worksheet_name, service_account = _load_settings()
        if not enabled:
            _READY = True
            return

        if not spreadsheet_id or not service_account:
            logger.warning("Google Sheets cache enabled but missing spreadsheet id or service account.")
            _READY = True
            return

        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception as e:
            logger.warning("Google Sheets cache disabled: missing dependencies (%s).", e)
            _READY = True
            return

        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            creds = Credentials.from_service_account_info(service_account, scopes=scopes)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(spreadsheet_id)
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=3)
                worksheet.update("A1:C1", [["cache_key", "value", "created_at"]])

            rows = worksheet.get_all_values()
            loaded = 0
            if rows:
                for row in rows[1:]:
                    if not row:
                        continue
                    cache_key = row[0].strip() if len(row) >= 1 else ""
                    if not cache_key:
                        continue
                    cache_val = row[1] if len(row) >= 2 else ""
                    _CACHE_MAP[cache_key] = cache_val
                    loaded += 1

            _WORKSHEET = worksheet
            _ENABLED = True
            logger.info("Google Sheets cache enabled (%s), loaded %s entries.", worksheet_name, loaded)
        except Exception as e:
            logger.warning("Failed to initialize Google Sheets cache: %s", e)
            _ENABLED = False
            _WORKSHEET = None
        finally:
            _READY = True


def cache_get(cache_key: str) -> Optional[str]:
    if not cache_key:
        return None
    _ensure_ready()
    if not _ENABLED:
        return None
    with _LOCK:
        return _CACHE_MAP.get(cache_key)


def cache_set(cache_key: str, value: str) -> None:
    if not cache_key or value is None:
        return
    _ensure_ready()
    if not _ENABLED:
        return

    with _LOCK:
        if cache_key in _CACHE_MAP:
            return
        if _WORKSHEET is None:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            _WORKSHEET.append_row([cache_key, str(value), now], value_input_option="RAW")
            _CACHE_MAP[cache_key] = str(value)
        except Exception as e:
            logger.warning("Google Sheets cache write failed: %s", e)
