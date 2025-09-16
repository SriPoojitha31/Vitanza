from fastapi import APIRouter, Depends
from models import GISData
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from routes.auth import require_roles

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.gis_db if mongo_client else None

@router.get("/heatmap", response_model=list[GISData])
async def get_heatmap(user=Depends(require_roles("admin", "officer"))):
    if db:
        data = await db.cases.find().to_list(100)
        return data
    # Fallback heatmap demo points
    return [
        {"lat": 26.85, "lon": 94.20, "intensity": 0.7, "timestamp": datetime.utcnow().isoformat()},
        {"lat": 26.55, "lon": 93.45, "intensity": 0.4, "timestamp": datetime.utcnow().isoformat()},
        {"lat": 27.05, "lon": 94.55, "intensity": 0.9, "timestamp": datetime.utcnow().isoformat()},
    ]