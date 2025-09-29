from fastapi import APIRouter, Depends
from models import WaterQualityReport
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from routes.auth_mongo import require_roles
from ml.predictor import OutbreakRiskPredictor
import pandas as pd
from pydantic import BaseModel
from fastapi import UploadFile, File, Form
import io, csv

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
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

class SensorIn(BaseModel):
    sensor_id: str
    ph: float
    turbidity: float
    tds: float
    temp: float
    location: str | None = None
    lat: float | None = None
    lon: float | None = None

_predictor = OutbreakRiskPredictor()

@router.post("/", response_model=WaterQualityReport)
async def submit_water_report(report: WaterQualityReport, user=Depends(require_roles("admin", "officer", "worker", "community"))):
    if db:
        await db.reports.insert_one(report.dict())
        # continue to enrich with risk below
    else:
        _water_reports_fallback.append(report.dict())
    # ML risk using predictor (ph, turbidity, optional tds/temp if present on the model input)
    df = pd.DataFrame([[
        getattr(report, 'ph', None),
        getattr(report, 'turbidity', None),
        getattr(report, 'tds', 250.0),
        getattr(report, 'temp', 25.0)
    ]], columns=["ph","turbidity","tds","temp"])
    proba = _predictor.predict_proba(df)[0]
    pred = int(proba[1] >= 0.3)  # Lower threshold for more accurate predictions
    risk = "high" if pred == 1 else "low"
    enriched = report.dict() | {"risk": risk, "risk_proba": float(proba[1])}
    # Persist enriched record
    if db:
        await db.reports.update_one({
            "sensor_id": report.sensor_id,
            "timestamp": report.timestamp or enriched.get("timestamp")
        }, {"$set": enriched}, upsert=True)
        # Auto-create alert for high risk
        if risk == "high":
            await db.alerts.insert_one({
                "type": "water_quality_emergency",
                "severity": "high",
                "message": f"Unsafe water detected at {enriched.get('location')}",
                "payload": enriched,
                "timestamp": datetime.utcnow().isoformat()
            })
    return enriched
    _water_reports_fallback.append(report.dict())
    return report

@router.get("/", response_model=list[WaterQualityReport])
async def list_water_reports(user=Depends(require_roles("admin", "officer", "worker", "community"))):
    if db:
        reports = await db.reports.find().to_list(100)
    else:
        reports = _water_reports_fallback
    # Tag risk for existing records
    for r in reports:
        if "risk" not in r:
            r["risk"] = "high" if (r.get("turbidity",0)>5 or r.get("ph",7)<6.5 or r.get("ph",7)>8.5) else "low"
    return reports

@router.post("/upload")
async def upload_water_csv(file: UploadFile = File(...), user=Depends(require_roles("admin", "officer", "worker", "community"))):
    content = await file.read()
    text = content.decode(errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    processed = 0
    for row in reader:
        try:
            payload = {
                "sensor_id": row.get("sensor_id") or f"csv-{processed}",
                "ph": float(row.get("ph")),
                "turbidity": float(row.get("turbidity")),
                "tds": float(row.get("tds")) if row.get("tds") else 250.0,
                "temp": float(row.get("temp")) if row.get("temp") else 25.0,
                "location": row.get("location") or None,
                "timestamp": datetime.utcnow().isoformat()
            }
            if db:
                await db.reports.insert_one(payload)
                # Auto alert if unsafe thresholds
                ph = payload.get("ph", 7)
                turb = payload.get("turbidity", 0)
                if turb > 4.0 or ph < 6.5 or ph > 8.5:
                    await db.alerts.insert_one({
                        "type": "water_quality_emergency",
                        "severity": "high",
                        "message": f"Unsafe water detected at {payload.get('location')}",
                        "payload": payload,
                        "timestamp": payload["timestamp"]
                    })
            else:
                _water_reports_fallback.append(payload)
            processed += 1
        except Exception:
            continue
    return {"success": True, "processedCount": processed}