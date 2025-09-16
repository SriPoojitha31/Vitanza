from fastapi import APIRouter, Depends
from models import HealthReport
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from routes.auth import require_roles

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
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
async def submit_health_report(report: HealthReport, user=Depends(require_roles("admin", "officer", "worker"))):
    if db:
        await db.reports.insert_one(report.dict())
        return report
    _health_reports_fallback.append(report.dict())
    return report

@router.get("/", response_model=list[HealthReport])
async def list_health_reports(user=Depends(require_roles("admin", "officer"))):
    if db:
        reports = await db.reports.find().to_list(100)
        return reports
    return _health_reports_fallback