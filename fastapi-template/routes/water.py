from fastapi import APIRouter
from schemas import WaterQualityReport
from motor.motor_asyncio import AsyncIOMotorClient
import os

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.water_db

@router.post("/", response_model=WaterQualityReport)
async def submit_water_report(report: WaterQualityReport):
    await db.reports.insert_one(report.dict())
    return report

@router.get("/", response_model=list[WaterQualityReport])
async def list_water_reports():
    reports = await db.reports.find().to_list(100)
    return reports