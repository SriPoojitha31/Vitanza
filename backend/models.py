from typing import Optional, List
from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserOut(BaseModel):
    id: int
    username: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class HealthReport(BaseModel):
    patient_id: str
    symptoms: List[str]
    location: str
    timestamp: str

class WaterQualityReport(BaseModel):
    sensor_id: str
    ph: float
    turbidity: float
    location: str
    timestamp: str

class Alert(BaseModel):
    message: str
    severity: str
    location: Optional[str]
    timestamp: str

class Feedback(BaseModel):
    user_id: str
    message: str
    rating: Optional[int]
    timestamp: str

class AwarenessContent(BaseModel):
    key: str
    lang: str
    title: str
    body: str

class GISData(BaseModel):
    lat: float
    lon: float
    intensity: float
    timestamp: str

class BatchSyncRequest(BaseModel):
    reports: List[HealthReport]