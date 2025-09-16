import os
from typing import Literal
import requests

# Simple SMS sender using a generic gateway (configure via env)
SMS_GATEWAY_URL = os.getenv("SMS_GATEWAY_URL")
SMS_API_KEY = os.getenv("SMS_API_KEY")
SMS_FAKE_MODE = os.getenv("SMS_FAKE_MODE", "true").lower() == "true"

LANG_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "alert": "ALERT [{severity}] {message}",
    },
    "hi": {
        "alert": "चेतावनी [{severity}] {message}",
    },
    "te": {
        "alert": "హెచ్చరిక [{severity}] {message}",
    },
}

async def send_firebase_alert(message: str, severity: str):
    # Placeholder stub for push notifications
    return True

def format_alert_message(message: str, severity: str, lang: str = "en") -> str:
    templates = LANG_TEMPLATES.get(lang, LANG_TEMPLATES["en"]) 
    return templates["alert"].format(severity=severity.upper(), message=message)

def send_sms_alert(phone: str, message: str, severity: str, lang: str = "en") -> bool:
    # If no gateway configured, simulate success to enable demo flows
    if SMS_FAKE_MODE or not SMS_GATEWAY_URL or not SMS_API_KEY:
        return True
    payload = {
        "api_key": SMS_API_KEY,
        "to": phone,
        "message": format_alert_message(message, severity, lang),
    }
    try:
        r = requests.post(SMS_GATEWAY_URL, json=payload, timeout=10)
        return r.status_code in (200, 201)
    except Exception:
        return False

