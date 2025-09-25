from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional, List
from bson import ObjectId
from mongo import get_mongo_db
from routes.auth_mongo import require_roles

router = APIRouter()

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class Community(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str
    district: Optional[str] = None
    state: Optional[str] = None
    population: Optional[int] = None

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

@router.get("/", response_model=List[Community])
async def list_communities(db=Depends(get_mongo_db)):
    cursor = db.communities.find({}, limit=200).sort("name")
    items = [Community(**doc) async for doc in cursor]
    return items

@router.post("/", response_model=Community)
async def create_community(payload: Community, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "worker", "community"))):
    data = payload.dict(by_alias=True, exclude_unset=True)
    res = await db.communities.insert_one(data)
    created = await db.communities.find_one({"_id": res.inserted_id})
    item = Community(**created)
    await manager.broadcast({"type": "created", "data": item.model_dump(by_alias=True)})
    return item

@router.get("/{community_id}", response_model=Community)
async def get_community(community_id: str, db=Depends(get_mongo_db)):
    doc = await db.communities.find_one({"_id": ObjectId(community_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Community not found")
    return Community(**doc)

@router.put("/{community_id}", response_model=Community)
async def update_community(community_id: str, payload: Community, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "worker", "community"))):
    data = payload.dict(by_alias=True, exclude_unset=True)
    await db.communities.update_one({"_id": ObjectId(community_id)}, {"$set": data})
    doc = await db.communities.find_one({"_id": ObjectId(community_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Community not found")
    item = Community(**doc)
    await manager.broadcast({"type": "updated", "data": item.model_dump(by_alias=True)})
    return item

@router.delete("/{community_id}")
async def delete_community(community_id: str, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "worker", "community"))):
    res = await db.communities.delete_one({"_id": ObjectId(community_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Community not found")
    await manager.broadcast({"type": "deleted", "id": community_id})
    return {"deleted": True}

# --- WebSocket for realtime updates ---
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()

@router.websocket("/ws")
async def communities_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive/no-op
    except WebSocketDisconnect:
        manager.disconnect(websocket)


