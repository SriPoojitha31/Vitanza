from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    WATER_QUALITY = "water_quality"
    AIR_QUALITY = "air_quality"
    PRESSURE = "pressure"
    PH = "ph"
    TURBIDITY = "turbidity"
    CHLORINE = "chlorine"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SensorData(BaseModel):
    device_id: str = Field(..., description="Unique identifier for the hardware device")
    sensor_type: SensorType = Field(..., description="Type of sensor reading")
    value: float = Field(..., description="Sensor reading value")
    unit: str = Field(..., description="Unit of measurement (e.g., '°C', '%', 'ppm')")
    location: Optional[str] = Field(None, description="Geographic location of the sensor")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of the reading")
    battery_level: Optional[float] = Field(None, description="Battery level percentage")
    signal_strength: Optional[int] = Field(None, description="Signal strength in dBm")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional sensor metadata")

class MultiSensorData(BaseModel):
    device_id: str = Field(..., description="Unique identifier for the hardware device")
    readings: List[SensorData] = Field(..., description="Multiple sensor readings from the same device")
    location: Optional[str] = Field(None, description="Geographic location of the device")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of the readings")
    device_status: Optional[str] = Field("online", description="Device status (online, offline, maintenance)")

class AlertData(BaseModel):
    device_id: str = Field(..., description="Device that triggered the alert")
    alert_type: str = Field(..., description="Type of alert (sensor_failure, threshold_exceeded, etc.)")
    severity: SeverityLevel = Field(..., description="Severity level of the alert")
    message: str = Field(..., description="Human-readable alert message")
    sensor_data: Optional[SensorData] = Field(None, description="Associated sensor data")
    location: Optional[str] = Field(None, description="Location where alert occurred")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Alert timestamp")

class DeviceStatus(BaseModel):
    device_id: str = Field(..., description="Device identifier")
    status: str = Field(..., description="Device status")
    last_seen: datetime = Field(..., description="Last communication timestamp")
    battery_level: Optional[float] = Field(None, description="Battery level")
    signal_strength: Optional[int] = Field(None, description="Signal strength")
    firmware_version: Optional[str] = Field(None, description="Firmware version")
    location: Optional[str] = Field(None, description="Device location")

class CommandData(BaseModel):
    device_id: str = Field(..., description="Target device identifier")
    command: str = Field(..., description="Command to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Command parameters")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Command timestamp")

class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type (sensor_data, alert, command, status)")
    data: Dict[str, Any] = Field(..., description="Message payload")
    device_id: Optional[str] = Field(None, description="Source device identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Message timestamp")
