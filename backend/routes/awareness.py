from fastapi import APIRouter, Depends
from models import AwarenessContent
from crud import get_awareness_content
from sqlalchemy.orm import Session
from routes.auth import require_roles
from typing import List

router = APIRouter()

def get_db():
    # Implement your DB session retrieval here
    pass

@router.get("/{lang}/{key}", response_model=AwarenessContent)
async def get_content(lang: str, key: str, db: Session = Depends(get_db), user=Depends(require_roles("admin", "officer", "worker"))):
    content = get_awareness_content(db, lang, key)
    if not content:
        # fallback demo content
        return {"key": key, "lang": lang, "title": "Safe Water Practices", "body": "Boil water, wash hands, and use clean storage."}
    return content

_awareness_fallback: list[dict] = [
    {"key": "hygiene", "lang": "en", "title": "Hygiene Tips", "body": "Wash hands, keep surroundings clean."},
    {"key": "boil", "lang": "en", "title": "Boil Water", "body": "Boil for 10 minutes before drinking."},
]

@router.get("/list/{lang}", response_model=List[AwarenessContent])
async def list_content(lang: str, user=Depends(require_roles("admin", "officer", "worker"))):
    return [c for c in _awareness_fallback if c["lang"] == lang] or _awareness_fallback