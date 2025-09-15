from fastapi import APIRouter, Depends
from schemas import AwarenessContent
from crud import get_awareness_content
from sqlalchemy.orm import Session

router = APIRouter()

def get_db():
    # Implement your DB session retrieval here
    pass

@router.get("/{lang}/{key}", response_model=AwarenessContent)
async def get_content(lang: str, key: str, db: Session = Depends(get_db)):
    content = get_awareness_content(db, lang, key)
    if not content:
        return {"key": key, "lang": lang, "title": "Not found", "body": ""}
    return content