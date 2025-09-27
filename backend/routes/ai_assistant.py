"""
AI Assistant API Routes
======================

This module provides FastAPI routes for AI assistant functionality
including symptom analysis, outbreak explanations, and health awareness.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ml.ai_assistant import (
    get_ai_assistant, analyze_symptoms, explain_outbreak_prediction,
    create_health_awareness_message, chat_with_agent
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SymptomAnalysisRequest(BaseModel):
    symptoms: str
    language: str = "English"
    patient_id: Optional[str] = None
    location: Optional[str] = None

class SymptomAnalysisResponse(BaseModel):
    analysis: str
    language: str
    timestamp: str
    symptoms: str
    recommendations: List[str]
    severity_level: str
    water_borne_risk: bool

class OutbreakExplanationRequest(BaseModel):
    prediction_data: Dict[str, Any]
    language: str = "English"
    community: Optional[str] = None

class OutbreakExplanationResponse(BaseModel):
    explanation: str
    language: str
    timestamp: str
    risk_level: str
    preventive_measures: List[str]
    community_actions: List[str]

class HealthAwarenessRequest(BaseModel):
    topic: str
    community: str
    language: str = "English"
    audience: str = "general"

class HealthAwarenessResponse(BaseModel):
    message: str
    topic: str
    community: str
    language: str
    audience: str
    timestamp: str

class ChatRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    query: str
    confidence: float
    sources: List[str]

@router.post("/analyze-symptoms", response_model=SymptomAnalysisResponse)
async def analyze_symptoms_endpoint(request: SymptomAnalysisRequest) -> SymptomAnalysisResponse:
    """
    Analyze symptoms and provide health advice.
    
    This endpoint uses AI to analyze reported symptoms and provide
    multilingual health advice, including water-borne disease risk assessment.
    """
    try:
        # Get AI assistant
        assistant = get_ai_assistant()
        
        # Analyze symptoms
        result = assistant.analyze_symptoms(request.symptoms, request.language)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract additional information
        analysis = result.get("analysis", "")
        
        # Determine severity level based on keywords
        severity_keywords = {
            "critical": ["severe", "critical", "emergency", "urgent", "acute"],
            "high": ["high", "serious", "concerning", "worrying"],
            "medium": ["moderate", "mild", "some"],
            "low": ["minor", "slight", "little"]
        }
        
        severity_level = "medium"  # default
        for level, keywords in severity_keywords.items():
            if any(keyword in analysis.lower() for keyword in keywords):
                severity_level = level
                break
        
        # Check for water-borne disease indicators
        water_borne_keywords = [
            "diarrhea", "vomiting", "fever", "cholera", "dysentery", 
            "typhoid", "contaminated water", "water-borne"
        ]
        water_borne_risk = any(keyword in request.symptoms.lower() for keyword in water_borne_keywords)
        
        # Generate recommendations
        recommendations = [
            "Seek medical attention if symptoms worsen",
            "Maintain proper hydration",
            "Practice good hygiene",
            "Avoid contaminated water sources"
        ]
        
        if water_borne_risk:
            recommendations.extend([
                "Use only safe, treated water",
                "Boil water before consumption",
                "Report to health authorities"
            ])
        
        return SymptomAnalysisResponse(
            analysis=analysis,
            language=request.language,
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            symptoms=request.symptoms,
            recommendations=recommendations,
            severity_level=severity_level,
            water_borne_risk=water_borne_risk
        )
        
    except Exception as e:
        logger.error(f"Symptom analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Symptom analysis failed: {str(e)}")

@router.post("/explain-outbreak", response_model=OutbreakExplanationResponse)
async def explain_outbreak_endpoint(request: OutbreakExplanationRequest) -> OutbreakExplanationResponse:
    """
    Explain outbreak prediction results in simple terms.
    
    This endpoint provides human-readable explanations of ML model
    predictions for community health workers and officials.
    """
    try:
        # Get AI assistant
        assistant = get_ai_assistant()
        
        # Explain outbreak prediction
        result = assistant.explain_outbreak_prediction(
            request.prediction_data, 
            request.language
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        explanation = result.get("explanation", "")
        
        # Extract risk level from prediction data
        risk_level = "medium"
        if "outbreak_probability" in request.prediction_data:
            prob = request.prediction_data["outbreak_probability"]
            if prob >= 0.8:
                risk_level = "critical"
            elif prob >= 0.6:
                risk_level = "high"
            elif prob >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
        
        # Generate preventive measures
        preventive_measures = [
            "Increase water quality monitoring",
            "Enhance sanitation measures",
            "Conduct health screenings",
            "Prepare emergency response"
        ]
        
        if risk_level in ["critical", "high"]:
            preventive_measures.extend([
                "Activate emergency protocols",
                "Deploy additional health workers",
                "Implement containment measures",
                "Alert regional health authorities"
            ])
        
        # Generate community actions
        community_actions = [
            "Inform community leaders",
            "Conduct awareness campaigns",
            "Distribute health information",
            "Monitor high-risk individuals"
        ]
        
        return OutbreakExplanationResponse(
            explanation=explanation,
            language=request.language,
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            risk_level=risk_level,
            preventive_measures=preventive_measures,
            community_actions=community_actions
        )
        
    except Exception as e:
        logger.error(f"Outbreak explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Outbreak explanation failed: {str(e)}")

@router.post("/create-awareness", response_model=HealthAwarenessResponse)
async def create_awareness_endpoint(request: HealthAwarenessRequest) -> HealthAwarenessResponse:
    """
    Create health awareness messages for communities.
    
    This endpoint generates culturally appropriate health awareness
    messages in multiple languages for different audiences.
    """
    try:
        # Get AI assistant
        assistant = get_ai_assistant()
        
        # Create awareness message
        result = assistant.create_health_awareness_message(
            request.topic,
            request.community,
            request.language,
            request.audience
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return HealthAwarenessResponse(
            message=result.get("message", ""),
            topic=request.topic,
            community=request.community,
            language=request.language,
            audience=request.audience,
            timestamp=result.get("timestamp", datetime.utcnow().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Health awareness creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health awareness creation failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Chat with the AI assistant using all available tools.
    
    This endpoint provides conversational AI assistance with access to
    translation, database, sensor, and health advice tools.
    """
    try:
        # Get AI assistant
        assistant = get_ai_assistant()
        
        # Process chat query
        result = assistant.chat_with_agent(request.query)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        response = result.get("response", "")
        
        # Calculate confidence (simple heuristic)
        confidence = 0.8 if len(response) > 50 else 0.6
        
        # Extract sources (if any)
        sources = []
        if "database" in response.lower():
            sources.append("Database")
        if "sensor" in response.lower():
            sources.append("IoT Sensors")
        if "translation" in response.lower():
            sources.append("Translation Service")
        if "health" in response.lower():
            sources.append("Health Knowledge Base")
        
        return ChatResponse(
            response=response,
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            query=request.query,
            confidence=confidence,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.get("/assistant/status")
async def get_assistant_status() -> Dict[str, Any]:
    """
    Get AI assistant status and capabilities.
    
    Returns:
    --------
    Dict[str, Any]
        Assistant status information
    """
    try:
        assistant = get_ai_assistant()
        status = assistant.get_assistant_info()
        
        return {
            "status": "healthy" if status["llm_loaded"] else "degraded",
            "capabilities": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get assistant status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assistant status: {str(e)}")

@router.post("/assistant/reload")
async def reload_assistant(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Reload the AI assistant models.
    
    Returns:
    --------
    Dict[str, str]
        Reload status
    """
    try:
        # Reload assistant in background
        background_tasks.add_task(_reload_assistant)
        
        return {"status": "Assistant reload initiated"}
        
    except Exception as e:
        logger.error(f"Assistant reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assistant reload failed: {str(e)}")

def _reload_assistant():
    """Background task to reload assistant."""
    try:
        global _ai_assistant
        _ai_assistant = None
        logger.info("AI assistant reloaded")
    except Exception as e:
        logger.error(f"Background assistant reload failed: {e}")

@router.get("/languages/supported")
async def get_supported_languages() -> Dict[str, List[str]]:
    """
    Get list of supported languages for AI assistant.
    
    Returns:
    --------
    Dict[str, List[str]]
        Supported languages by feature
    """
    return {
        "symptom_analysis": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
        "outbreak_explanation": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
        "health_awareness": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
        "chat": ["English", "Bengali", "Hindi", "Telugu", "Assamese"]
    }

@router.get("/topics/health")
async def get_health_topics() -> List[Dict[str, str]]:
    """
    Get list of available health topics for awareness messages.
    
    Returns:
    --------
    List[Dict[str, str]]
        Available health topics
    """
    return [
        {"id": "water_safety", "name": "Water Safety", "description": "Safe water practices and contamination prevention"},
        {"id": "hygiene", "name": "Personal Hygiene", "description": "Hand washing and sanitation practices"},
        {"id": "disease_prevention", "name": "Disease Prevention", "description": "Preventing common water-borne diseases"},
        {"id": "nutrition", "name": "Nutrition", "description": "Healthy eating and nutrition guidelines"},
        {"id": "vaccination", "name": "Vaccination", "description": "Importance of vaccinations and immunization"},
        {"id": "emergency_preparedness", "name": "Emergency Preparedness", "description": "Preparing for health emergencies"},
        {"id": "maternal_health", "name": "Maternal Health", "description": "Prenatal and postnatal care"},
        {"id": "child_health", "name": "Child Health", "description": "Child health and development"}
    ]

@router.get("/audiences")
async def get_target_audiences() -> List[Dict[str, str]]:
    """
    Get list of target audiences for health messages.
    
    Returns:
    --------
    List[Dict[str, str]]
        Target audiences
    """
    return [
        {"id": "general", "name": "General Public", "description": "General community members"},
        {"id": "health_workers", "name": "Health Workers", "description": "Community health workers and nurses"},
        {"id": "children", "name": "Children", "description": "School-age children and adolescents"},
        {"id": "elderly", "name": "Elderly", "description": "Senior citizens and elderly community members"},
        {"id": "pregnant_women", "name": "Pregnant Women", "description": "Expecting mothers and new mothers"},
        {"id": "leaders", "name": "Community Leaders", "description": "Village leaders and decision makers"}
    ]
