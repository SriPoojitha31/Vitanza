from fastapi import APIRouter, Depends
from models import Alert
from utils.notifications import send_firebase_alert, send_sms_alert
from datetime import datetime
from routes.auth import require_roles

router = APIRouter()

_alerts_memory: list[dict] = []

@router.post("/", response_model=Alert)
async def send_alert(alert: Alert, user=Depends(require_roles("admin", "officer"))):
    await send_firebase_alert(alert.message, alert.severity)
    # Ensure timestamp exists for clients
    if not getattr(alert, "timestamp", None):
        alert.timestamp = datetime.utcnow().isoformat()
    payload = alert.dict()
    _alerts_memory.append(payload)
    return alert

@router.post("/sms")
def send_alert_sms(message: str, severity: str, phone: str, lang: str = "en", user=Depends(require_roles("admin", "officer"))):
    ok = send_sms_alert(phone, message, severity, lang)
    return {"sent": ok}

@router.get("/")
def list_alerts(user=Depends(require_roles("admin", "officer"))):
    return _alerts_memory[-100:]