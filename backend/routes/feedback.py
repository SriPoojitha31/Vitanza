from fastapi import APIRouter, Depends
from models import Feedback
from motor.motor_asyncio import AsyncIOMotorClient
import os
from routes.auth import require_roles

router = APIRouter()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.feedback_db if mongo_client else None
_feedback_fallback: list[dict] = []

@router.post("/", response_model=Feedback)
async def submit_feedback(feedback: Feedback, user=Depends(require_roles("admin", "officer", "worker"))):
    if db:
        await db.feedback.insert_one(feedback.dict())
        return feedback
    _feedback_fallback.append(feedback.dict())
    return feedback

@router.get("/", response_model=list[Feedback])
async def list_feedback(user=Depends(require_roles("admin", "officer"))):
    if db:
        feedbacks = await db.feedback.find().to_list(100)
        return feedbacks
    return _feedback_fallback