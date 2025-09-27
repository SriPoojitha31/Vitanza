import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Vitanza")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from routes import (
    health, water, alerts, feedback, awareness, gis, email
)
from routes import auth_mongo
from routes import community
from routes import hardware
from routes import ml_inference
from routes import ai_assistant

app.include_router(auth_mongo.router, prefix="/api/auth", tags=["Auth"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(water.router, prefix="/api/water", tags=["Water"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(awareness.router, prefix="/api/awareness", tags=["Awareness"])
app.include_router(gis.router, prefix="/api/gis", tags=["GIS"])
app.include_router(community.router, prefix="/api/communities", tags=["Communities"])
app.include_router(email.router, prefix="/api/email", tags=["Email"])
app.include_router(hardware.router, prefix="/api/hardware", tags=["Hardware"])
app.include_router(ml_inference.router, prefix="/api/ml", tags=["ML Inference"])
app.include_router(ai_assistant.router, prefix="/api/ai", tags=["AI Assistant"])
# Emergency routes are optional; include only if present
try:
    from routes import emergency
    if hasattr(emergency, "router"):
        app.include_router(emergency.router, prefix="/api", tags=["Emergency"])
except Exception:
    pass

# Offline routes are optional; include only if present
try:
    from routes import offline
    if hasattr(offline, "router"):
        app.include_router(offline.router, prefix="/api/offline", tags=["Offline"])
except Exception:
    pass

# RBAC route groups
from routes import rbac
app.include_router(rbac.router, prefix="/api", tags=["RBAC"])

# Blockchain integrity logs
from utils import blockchain
app.include_router(blockchain.router, prefix="/api/blockchain", tags=["Blockchain"])

# Initialize MQTT subscriber for hardware communication
@app.on_event("startup")
async def startup_event():
    """Initialize MQTT subscriber on startup."""
    try:
        from utils.mqtt_client import initialize_mqtt_subscriber
        await initialize_mqtt_subscriber()
        print("MQTT subscriber initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MQTT subscriber: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up MQTT subscriber on shutdown."""
    try:
        from utils.mqtt_client import mqtt_subscriber
        if mqtt_subscriber:
            mqtt_subscriber.stop_loop()
            print("MQTT subscriber stopped")
    except Exception as e:
        print(f"Error stopping MQTT subscriber: {e}")

@app.get("/health-check", tags=["System"])
async def health_check():
    return {"status": "ok"}