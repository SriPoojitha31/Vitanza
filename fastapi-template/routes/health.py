from fastapi import APIRouter, Depends
from schemas import HealthReport
from motor.motor_asyncio import AsyncIOMotorClient
import os

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.health_db

@router.post("/", response_model=HealthReport)
async def submit_health_report(report: HealthReport):
    await db.reports.insert_one(report.dict())
    return report

@router.get("/", response_model=list[HealthReport])
async def list_health_reports():
    reports = await db.reports.find().to_list(100)
    return reports