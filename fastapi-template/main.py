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
    health, water, alerts, feedback, awareness, gis, offline, auth
)

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(water.router, prefix="/api/water", tags=["Water"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(awareness.router, prefix="/api/awareness", tags=["Awareness"])
app.include_router(gis.router, prefix="/api/gis", tags=["GIS"])
app.include_router(offline.router, prefix="/api/offline", tags=["Offline"])

@app.get("/health-check", tags=["System"])
async def health_check():
    return {"status": "ok"}