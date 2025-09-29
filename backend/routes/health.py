from fastapi import APIRouter, Depends, Form
from models import HealthReport
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from routes.auth_mongo import require_roles
from typing import Optional

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.health_db if mongo_client else None

# Basic in-memory fallback store
_health_reports_fallback: list[dict] = [
    {
        "patient_id": "demo-001",
        "symptoms": ["fever", "diarrhea"],
        "location": "Village A",
        "timestamp": datetime.utcnow().isoformat(),
    }
]

@router.post("/", response_model=HealthReport)
async def submit_health_report(report: HealthReport, user=Depends(require_roles("admin", "officer", "worker", "community"))):
    if db:
        await db.reports.insert_one(report.dict())
        return report
    _health_reports_fallback.append(report.dict())
    return report

@router.get("/", response_model=list[HealthReport])
async def list_health_reports(user=Depends(require_roles("admin", "officer", "worker", "community"))):
    if db:
        reports = await db.reports.find().to_list(100)
        return reports
    return _health_reports_fallback

@router.post("/sms")
async def ingest_sms(
    patient_id: str = Form(...),
    symptoms: str = Form(...),  # comma-separated
    location: str = Form("Unknown"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    user=Depends(require_roles("admin", "officer", "worker", "asha"))
):
    doc = {
        "patient_id": patient_id,
        "symptoms": [s.strip() for s in symptoms.split(',') if s.strip()],
        "location": location,
        "timestamp": datetime.utcnow().isoformat(),
        "lat": lat,
        "lon": lon,
    }
    if db:
        await db.reports.insert_one(doc)
    else:
        _health_reports_fallback.append(doc)
    return {"ok": True}