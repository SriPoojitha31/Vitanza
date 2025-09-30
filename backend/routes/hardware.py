import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Import our schemas
from schemas.hardware import (
    SensorData, MultiSensorData, AlertData, DeviceStatus, 
    CommandData, WebSocketMessage, SensorType, SeverityLevel
)

# Import database connection (assuming MongoDB)
from mongo import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for WebSocket connections and device status
active_connections: List[WebSocket] = []
device_connections: Dict[str, WebSocket] = {}
device_status: Dict[str, DeviceStatus] = {}

# ============================================================================
# HTTP ENDPOINTS (Method 1: Simple HTTP Requests)
# ============================================================================

@router.post("/sensor-data", response_model=Dict[str, str])
async def receive_sensor_data(data: SensorData):
    """
    Receive sensor data via HTTP POST request.
    Hardware sends JSON data to this endpoint.
    """
    try:
        # Log the received data
        logger.info(f"Received sensor data from device {data.device_id}: {data.sensor_type} = {data.value} {data.unit}")
        
        # Store in database
        db = get_db()
        collection = db["sensor_readings"]
        
        # Convert to dict for MongoDB storage
        sensor_dict = data.dict()
        sensor_dict["received_at"] = datetime.utcnow()
        
        result = await collection.insert_one(sensor_dict)
        
        # Check for threshold alerts
        await check_sensor_thresholds(data)
        
        # Broadcast to WebSocket connections if any
        await broadcast_sensor_data(data)
        
        return {"status": "received", "id": str(result.inserted_id)}
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        raise HTTPException(status_code=500, detail="Failed to process sensor data")

@router.post("/multi-sensor-data", response_model=Dict[str, str])
async def receive_multi_sensor_data(data: MultiSensorData):
    """
    Receive multiple sensor readings from a single device.
    """
    try:
        logger.info(f"Received multi-sensor data from device {data.device_id}: {len(data.readings)} readings")
        
        db = get_db()
        collection = db["sensor_readings"]
        
        # Store each reading
        readings_to_insert = []
        for reading in data.readings:
            reading_dict = reading.dict()
            reading_dict["device_id"] = data.device_id
            reading_dict["location"] = data.location
            reading_dict["latitude"] = data.latitude
            reading_dict["longitude"] = data.longitude
            reading_dict["received_at"] = datetime.utcnow()
            readings_to_insert.append(reading_dict)
            
            # Check thresholds for each reading
            await check_sensor_thresholds(reading)
        
        result = await collection.insert_many(readings_to_insert)
        
        # Broadcast to WebSocket connections
        await broadcast_multi_sensor_data(data)
        
        return {"status": "received", "count": len(readings_to_insert), "ids": [str(id) for id in result.inserted_ids]}
        
    except Exception as e:
        logger.error(f"Error processing multi-sensor data: {e}")
        raise HTTPException(status_code=500, detail="Failed to process sensor data")

@router.post("/alert", response_model=Dict[str, str])
async def receive_alert(alert: AlertData):
    """
    Receive alert data from hardware devices.
    """
    try:
        logger.warning(f"Received alert from device {alert.device_id}: {alert.alert_type} - {alert.message}")
        
        db = get_db()
        collection = db["alerts"]
        
        alert_dict = alert.dict()
        alert_dict["received_at"] = datetime.utcnow()
        
        result = await collection.insert_one(alert_dict)
        
        # Broadcast alert to WebSocket connections
        await broadcast_alert(alert)
        
        return {"status": "alert_received", "id": str(result.inserted_id)}
        
    except Exception as e:
        logger.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to process alert")

@router.get("/sensor-data/{device_id}")
async def get_device_data(device_id: str, limit: int = 100):
    """
    Retrieve sensor data for a specific device.
    """
    try:
        db = get_db()
        collection = db["sensor_readings"]
        
        cursor = collection.find({"device_id": device_id}).sort("timestamp", -1).limit(limit)
        data = await cursor.to_list(length=limit)
        
        return {"device_id": device_id, "data": data}
        
    except Exception as e:
        logger.error(f"Error retrieving device data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve device data")

# ============================================================================
# WEBSOCKET ENDPOINTS (Method 2: Real-time Communication)
# ============================================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional communication.
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive message from client/hardware
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the message
            await process_websocket_message(websocket, message_data)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        # Remove from device connections if it was a device
        device_id = None
        for dev_id, conn in device_connections.items():
            if conn == websocket:
                device_id = dev_id
                break
        if device_id:
            del device_connections[device_id]
            logger.info(f"Device {device_id} disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@router.websocket("/ws/device/{device_id}")
async def device_websocket_endpoint(websocket: WebSocket, device_id: str):
    """
    Dedicated WebSocket endpoint for specific device communication.
    """
    await websocket.accept()
    device_connections[device_id] = websocket
    
    # Update device status
    device_status[device_id] = DeviceStatus(
        device_id=device_id,
        status="online",
        last_seen=datetime.utcnow()
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process device-specific message
            await process_device_message(device_id, message_data)
            
    except WebSocketDisconnect:
        if device_id in device_connections:
            del device_connections[device_id]
        if device_id in device_status:
            device_status[device_id].status = "offline"
        logger.info(f"Device {device_id} disconnected")
    except Exception as e:
        logger.error(f"Device WebSocket error for {device_id}: {e}")
        if device_id in device_connections:
            del device_connections[device_id]

# ============================================================================
# WEBSOCKET MESSAGE PROCESSING
# ============================================================================

async def process_websocket_message(websocket: WebSocket, message_data: Dict[str, Any]):
    """Process incoming WebSocket messages."""
    message_type = message_data.get("type", "unknown")
    
    if message_type == "sensor_data":
        # Handle sensor data via WebSocket
        sensor_data = SensorData(**message_data.get("data", {}))
        await receive_sensor_data(sensor_data)
        
    elif message_type == "alert":
        # Handle alert via WebSocket
        alert_data = AlertData(**message_data.get("data", {}))
        await receive_alert(alert_data)
        
    elif message_type == "device_status":
        # Handle device status update
        device_id = message_data.get("device_id")
        if device_id:
            device_status[device_id] = DeviceStatus(**message_data.get("data", {}))
            
    elif message_type == "ping":
        # Respond to ping
        await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
        
    else:
        logger.warning(f"Unknown message type: {message_type}")

async def process_device_message(device_id: str, message_data: Dict[str, Any]):
    """Process device-specific WebSocket messages."""
    message_type = message_data.get("type", "unknown")
    
    if message_type == "sensor_data":
        sensor_data = SensorData(**message_data.get("data", {}))
        await receive_sensor_data(sensor_data)
        
    elif message_type == "alert":
        alert_data = AlertData(**message_data.get("data", {}))
        await receive_alert(alert_data)
        
    elif message_type == "status_update":
        # Update device status
        status_data = message_data.get("data", {})
        device_status[device_id] = DeviceStatus(
            device_id=device_id,
            **status_data
        )

# ============================================================================
# BROADCAST FUNCTIONS
# ============================================================================

async def broadcast_sensor_data(sensor_data: SensorData):
    """Broadcast sensor data to all connected WebSocket clients."""
    message = {
        "type": "sensor_data",
        "data": sensor_data.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            # Remove broken connections
            if connection in active_connections:
                active_connections.remove(connection)

async def broadcast_multi_sensor_data(multi_data: MultiSensorData):
    """Broadcast multi-sensor data to all connected WebSocket clients."""
    message = {
        "type": "multi_sensor_data",
        "data": multi_data.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            if connection in active_connections:
                active_connections.remove(connection)

async def broadcast_alert(alert: AlertData):
    """Broadcast alert to all connected WebSocket clients."""
    message = {
        "type": "alert",
        "data": alert.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            if connection in active_connections:
                active_connections.remove(connection)

# ============================================================================
# THRESHOLD MONITORING
# ============================================================================

async def check_sensor_thresholds(sensor_data: SensorData):
    """Check if sensor reading exceeds thresholds and create alerts."""
    thresholds = {
        SensorType.TEMPERATURE: {"min": -10, "max": 50},
        SensorType.HUMIDITY: {"min": 0, "max": 100},
        SensorType.PH: {"min": 6.5, "max": 8.5},
        SensorType.TURBIDITY: {"min": 0, "max": 5.0},
        SensorType.CHLORINE: {"min": 0.2, "max": 4.0}
    }
    
    if sensor_data.sensor_type in thresholds:
        threshold = thresholds[sensor_data.sensor_type]
        
        if sensor_data.value < threshold["min"] or sensor_data.value > threshold["max"]:
            alert = AlertData(
                device_id=sensor_data.device_id,
                alert_type="threshold_exceeded",
                severity=SeverityLevel.HIGH if abs(sensor_data.value - threshold["min"]) > threshold["max"] * 0.5 else SeverityLevel.MEDIUM,
                message=f"{sensor_data.sensor_type} reading {sensor_data.value} {sensor_data.unit} exceeds threshold",
                sensor_data=sensor_data,
                location=sensor_data.location
            )
            
            # Store and broadcast the alert
            await receive_alert(alert)

# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

@router.get("/devices/status")
async def get_all_device_status():
    """Get status of all connected devices."""
    return {"devices": list(device_status.values())}

@router.get("/devices/{device_id}/status")
async def get_device_status(device_id: str):
    """Get status of a specific device."""
    if device_id not in device_status:
        raise HTTPException(status_code=404, detail="Device not found")
    return device_status[device_id]

@router.post("/devices/{device_id}/command")
async def send_device_command(device_id: str, command: CommandData):
    """Send a command to a specific device via WebSocket."""
    if device_id not in device_connections:
        raise HTTPException(status_code=404, detail="Device not connected")
    
    try:
        message = {
            "type": "command",
            "data": command.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await device_connections[device_id].send_text(json.dumps(message))
        return {"status": "command_sent", "device_id": device_id}
        
    except Exception as e:
        logger.error(f"Error sending command to device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send command")

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/connections")
async def get_connection_info():
    """Get information about active connections."""
    return {
        "active_connections": len(active_connections),
        "device_connections": len(device_connections),
        "connected_devices": list(device_connections.keys())
    }

@router.get("/health")
async def hardware_health_check():
    """Health check for hardware communication endpoints."""
    return {
        "status": "healthy",
        "websocket_connections": len(active_connections),
        "device_connections": len(device_connections),
        "timestamp": datetime.utcnow().isoformat()
    }
