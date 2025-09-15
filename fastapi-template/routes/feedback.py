from fastapi import APIRouter
from schemas import Feedback
from motor.motor_asyncio import AsyncIOMotorClient
import os

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.feedback_db

@router.post("/", response_model=Feedback)
async def submit_feedback(feedback: Feedback):
    await db.feedback.insert_one(feedback.dict())
    return feedback

@router.get("/", response_model=list[Feedback])
async def list_feedback():
    feedbacks = await db.feedback.find().to_list(100)
    return feedbacks