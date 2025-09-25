from fastapi import APIRouter, Depends
from models import Alert
from utils.notifications import send_firebase_alert, send_sms_alert
from datetime import datetime
from routes.auth_mongo import require_roles
from mongo import get_mongo_db

router = APIRouter()

@router.post("/", response_model=Alert)
async def send_alert(alert: Alert, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "government", "ngo", "asha"))):
    await send_firebase_alert(alert.message, alert.severity)
    if not getattr(alert, "timestamp", None):
        alert.timestamp = datetime.utcnow().isoformat()
    payload = alert.dict()
    await db.alerts.insert_one(payload)
    return alert

@router.post("/sms")
def send_alert_sms(message: str, severity: str, phone: str, lang: str = "en", user=Depends(require_roles("admin", "officer", "government", "ngo", "asha"))):
    ok = send_sms_alert(phone, message, severity, lang)
    return {"sent": ok}

@router.get("/")
async def list_alerts(db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "government", "ngo", "asha"))):
    items = await db.alerts.find({}, limit=200).sort("timestamp", -1).to_list(200)
    return items