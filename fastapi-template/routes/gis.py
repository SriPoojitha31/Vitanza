from fastapi import APIRouter
from schemas import GISData
from motor.motor_asyncio import AsyncIOMotorClient
import os

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.gis_db

@router.get("/heatmap", response_model=list[GISData])
async def get_heatmap():
    data = await db.cases.find().to_list(100)
    return data