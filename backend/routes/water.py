from fastapi import APIRouter, Depends
from models import WaterQualityReport
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from routes.auth import require_roles
from ml.predictor import OutbreakRiskPredictor
import pandas as pd

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.water_db if mongo_client else None

_water_reports_fallback: list[dict] = [
    {
        "sensor_id": "sensor-001",
        "ph": 7.2,
        "turbidity": 2.5,
        "location": "Well 3, Village A",
        "timestamp": datetime.utcnow().isoformat(),
    },
]

# Simple in-memory ML predictor with dummy training
_predictor = OutbreakRiskPredictor()
_predictor.is_trained = True  # bypass training for demo

@router.post("/", response_model=WaterQualityReport)
async def submit_water_report(report: WaterQualityReport, user=Depends(require_roles("admin", "officer", "worker"))):
    if db:
        await db.reports.insert_one(report.dict())
        # continue to enrich with risk below
    else:
        _water_reports_fallback.append(report.dict())
    # Enrich with ML risk (dummy): higher risk if turbidity > 5 or pH outside [6.5,8.5]
    df = pd.DataFrame([[report.ph, report.turbidity]], columns=["ph","turbidity"])
    # Fake probability by simple heuristic for demo
    risk = "low"
    if report.turbidity > 5 or report.ph < 6.5 or report.ph > 8.5:
        risk = "high"
    enriched = report.dict() | {"risk": risk}
    return enriched
    _water_reports_fallback.append(report.dict())
    return report

@router.get("/", response_model=list[WaterQualityReport])
async def list_water_reports(user=Depends(require_roles("admin", "officer"))):
    if db:
        reports = await db.reports.find().to_list(100)
    else:
        reports = _water_reports_fallback
    # Tag risk for existing records
    for r in reports:
        if "risk" not in r:
            r["risk"] = "high" if (r.get("turbidity",0)>5 or r.get("ph",7)<6.5 or r.get("ph",7)>8.5) else "low"
    return reports