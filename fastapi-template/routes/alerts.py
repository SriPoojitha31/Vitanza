from fastapi import APIRouter
from schemas import Alert
from utils.notifications import send_firebase_alert

router = APIRouter()

@router.post("/", response_model=Alert)
async def send_alert(alert: Alert):
    await send_firebase_alert(alert.message, alert.severity)
    return alert