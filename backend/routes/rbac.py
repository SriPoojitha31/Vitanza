from fastapi import APIRouter, Depends, HTTPException
from routes.auth_mongo import require_roles
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# Communities CRUD (demo in-memory)
class Community(BaseModel):
    id: int
    name: str
    district: str
    language: Optional[str] = "en"
    created_at: str

_communities: list[dict] = [
    {"id": 1, "name": "Village A", "district": "District 1", "language": "en", "created_at": datetime.utcnow().isoformat()},
    {"id": 2, "name": "Village B", "district": "District 2", "language": "hi", "created_at": datetime.utcnow().isoformat()},
]

@router.get("/community", response_model=List[Community])
async def list_communities(user=Depends(require_roles("admin", "officer", "worker", "community"))):
    return _communities

class CommunityCreate(BaseModel):
    name: str
    district: str
    language: Optional[str] = "en"

@router.post("/community", response_model=Community)
async def create_community(body: CommunityCreate, user=Depends(require_roles("admin", "officer"))):
    new = {
        "id": (max([c["id"] for c in _communities]) + 1) if _communities else 1,
        "name": body.name,
        "district": body.district,
        "language": body.language,
        "created_at": datetime.utcnow().isoformat(),
    }
    _communities.append(new)
    return new

@router.delete("/community/{community_id}")
async def delete_community(community_id: int, user=Depends(require_roles("admin"))):
    i = next((idx for idx, c in enumerate(_communities) if c["id"] == community_id), -1)
    if i == -1:
        raise HTTPException(status_code=404, detail="Not found")
    _communities.pop(i)
    return {"ok": True}

# Water summary endpoint for authority dashboard
class WaterSummary(BaseModel):
    safe: int
    caution: int
    unsafe: int

@router.get("/authority/water-summary", response_model=WaterSummary)
async def water_summary(user=Depends(require_roles("admin", "officer"))):
    # demo numbers matching landing page
    return {"safe": 142, "caution": 23, "unsafe": 8}

# Authority dashboard summary
class AuthoritySummary(BaseModel):
    active_monitors: int
    communities: int
    safe_areas_pct: float

@router.get("/authority/summary", response_model=AuthoritySummary)
async def authority_summary(user=Depends(require_roles("admin", "officer"))):
    return {"active_monitors": 2847, "communities": 156, "safe_areas_pct": 98.2}

# Admin user role management (demo; uses in-memory crud in auth file scope)
from crud import _users_db  # type: ignore

class UserRole(BaseModel):
    username: str
    role: str

@router.get("/admin/users")
async def list_users(user=Depends(require_roles("admin"))):
    return list(_users_db.values())

@router.post("/admin/users/role")
async def set_user_role(body: UserRole, user=Depends(require_roles("admin"))):
    u = _users_db.get(body.username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    u["role"] = body.role
    return u
