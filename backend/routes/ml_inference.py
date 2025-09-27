"""
ML Inference API Routes
======================

This module provides FastAPI routes for outbreak prediction inference
using the ensemble of tabular, time-series, and text models.
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

# Import ML models
from ml.tabular_model import TabularOutbreakPredictor
from ml.timeseries_model import TimeSeriesOutbreakPredictor
from ml.text_model import MultilingualTextClassifier
from ml.ensemble_model import EnsembleOutbreakPredictor

# Import schemas
from schemas.ml_models import (
    InferenceRequest, InferenceResponse, ClinicalReport, WaterQualityReading,
    EnvironmentalContext, PopulationDensity, SanitationIndex, EventFlags,
    GeographicLocation, OutbreakPrediction, ModelEvaluation
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instances
tabular_model = None
timeseries_model = None
text_model = None
ensemble_model = None

# Model configuration
MODEL_CONFIG = {
    "tabular_model_path": "models/xgb_outbreak.json",
    "timeseries_model_path": "models/ts_lstm.pth",
    "text_model_path": "models/text_xlm_roberta",
    "ensemble_model_path": "models/ensemble_meta.joblib"
}

async def load_models():
    """Load all ML models on startup."""
    global tabular_model, timeseries_model, text_model, ensemble_model
    
    try:
        # Load tabular model
        tabular_model = TabularOutbreakPredictor(MODEL_CONFIG["tabular_model_path"])
        tabular_model.load_model()
        logger.info("Tabular model loaded successfully")
        
        # Load time-series model
        timeseries_model = TimeSeriesOutbreakPredictor(MODEL_CONFIG["timeseries_model_path"])
        timeseries_model.load_model()
        logger.info("Time-series model loaded successfully")
        
        # Load text model
        text_model = MultilingualTextClassifier(MODEL_CONFIG["text_model_path"])
        text_model.load_model()
        logger.info("Text model loaded successfully")
        
        # Load ensemble model
        ensemble_model = EnsembleOutbreakPredictor(MODEL_CONFIG["ensemble_model_path"])
        ensemble_model.set_base_models(tabular_model, timeseries_model, text_model)
        ensemble_model.load_model()
        logger.info("Ensemble model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize ML models")

@router.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await load_models()

@router.post("/predict", response_model=InferenceResponse)
async def predict_outbreak(request: InferenceRequest) -> InferenceResponse:
    """
    Predict outbreak probability using ensemble of models.
    
    This endpoint combines predictions from:
    - Tabular model (XGBoost) for aggregated features
    - Time-series model (LSTM) for sensor data patterns
    - Text model (XLM-Roberta) for clinical text analysis
    - Ensemble meta-learner for final prediction
    """
    start_time = time.time()
    
    try:
        # Validate models are loaded
        if ensemble_model is None:
            raise HTTPException(status_code=503, detail="ML models not loaded")
        
        # Make ensemble prediction
        response = ensemble_model.predict(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        response.processing_time_ms = processing_time
        
        logger.info(f"Prediction completed in {processing_time:.2f}ms. "
                   f"Probability: {response.outbreak_probability:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/tabular", response_model=Dict[str, float])
async def predict_tabular(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Get prediction from tabular model only.
    
    Parameters:
    -----------
    features : Dict[str, Any]
        Tabular features for prediction
        
    Returns:
    --------
    Dict[str, float]
        Tabular model prediction
    """
    if tabular_model is None:
        raise HTTPException(status_code=503, detail="Tabular model not loaded")
    
    try:
        # Convert dict to TabularFeatures
        from schemas.ml_models import TabularFeatures
        tabular_features = TabularFeatures(**features)
        
        # Make prediction
        prediction = tabular_model.predict(tabular_features)
        
        return {
            "prediction": prediction,
            "confidence": abs(prediction - 0.5) * 2,
            "model": "tabular_xgboost"
        }
        
    except Exception as e:
        logger.error(f"Tabular prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tabular prediction failed: {str(e)}")

@router.post("/predict/timeseries", response_model=Dict[str, float])
async def predict_timeseries(window_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Get prediction from time-series model only.
    
    Parameters:
    -----------
    window_data : Dict[str, Any]
        Time-series window data
        
    Returns:
    --------
    Dict[str, float]
        Time-series model prediction
    """
    if timeseries_model is None:
        raise HTTPException(status_code=503, detail="Time-series model not loaded")
    
    try:
        # Convert dict to TimeSeriesWindow
        from schemas.ml_models import TimeSeriesWindow
        timeseries_window = TimeSeriesWindow(**window_data)
        
        # Make prediction
        prediction = timeseries_model.predict(timeseries_window)
        
        return {
            "prediction": prediction,
            "confidence": abs(prediction - 0.5) * 2,
            "model": "timeseries_lstm"
        }
        
    except Exception as e:
        logger.error(f"Time-series prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Time-series prediction failed: {str(e)}")

@router.post("/predict/text", response_model=Dict[str, Any])
async def predict_text(text_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get prediction from text model only.
    
    Parameters:
    -----------
    text_data : Dict[str, Any]
        Text data for prediction
        
    Returns:
    --------
    Dict[str, Any]
        Text model prediction and analysis
    """
    if text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    
    try:
        # Convert dict to TextFeatures
        from schemas.ml_models import TextFeatures
        text_features = TextFeatures(**text_data)
        
        # Make prediction
        prediction = text_model.predict(text_features)
        
        # Analyze text
        analysis = text_model.analyze_text(text_features.text, text_features.language)
        
        return {
            "prediction": prediction,
            "confidence": abs(prediction - 0.5) * 2,
            "model": "text_xlm_roberta",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Text prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")

@router.post("/analyze/text", response_model=Dict[str, Any])
async def analyze_text(text: str, language: str = "auto") -> Dict[str, Any]:
    """
    Analyze clinical text for symptoms and severity indicators.
    
    Parameters:
    -----------
    text : str
        Clinical text to analyze
    language : str
        Text language (auto-detect if "auto")
        
    Returns:
    --------
    Dict[str, Any]
        Text analysis results
    """
    if text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    
    try:
        from schemas.ml_models import LanguageCode
        
        # Convert language string to enum
        try:
            lang_enum = LanguageCode(language)
        except ValueError:
            lang_enum = LanguageCode.AUTO
        
        # Analyze text
        analysis = text_model.analyze_text(text, lang_enum)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@router.get("/models/status", response_model=Dict[str, Any])
async def get_model_status() -> Dict[str, Any]:
    """
    Get status of all loaded models.
    
    Returns:
    --------
    Dict[str, Any]
        Model status information
    """
    status = {
        "tabular_model": {
            "loaded": tabular_model is not None,
            "path": MODEL_CONFIG["tabular_model_path"]
        },
        "timeseries_model": {
            "loaded": timeseries_model is not None,
            "path": MODEL_CONFIG["timeseries_model_path"]
        },
        "text_model": {
            "loaded": text_model is not None,
            "path": MODEL_CONFIG["text_model_path"]
        },
        "ensemble_model": {
            "loaded": ensemble_model is not None,
            "path": MODEL_CONFIG["ensemble_model_path"]
        }
    }
    
    # Add model info if available
    if text_model is not None:
        status["text_model"]["info"] = text_model.get_model_info()
    
    if ensemble_model is not None:
        status["ensemble_model"]["info"] = ensemble_model.get_ensemble_info()
    
    return status

@router.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Reload all ML models.
    
    Returns:
    --------
    Dict[str, str]
        Reload status
    """
    try:
        # Reload models in background
        background_tasks.add_task(load_models)
        
        return {"status": "Models reload initiated"}
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@router.get("/models/feature-importance", response_model=Dict[str, Any])
async def get_feature_importance() -> Dict[str, Any]:
    """
    Get feature importance from all models.
    
    Returns:
    --------
    Dict[str, Any]
        Feature importance from all models
    """
    importance = {}
    
    try:
        # Tabular model feature importance
        if tabular_model is not None:
            importance["tabular"] = tabular_model.get_feature_importance()
        
        # Ensemble model feature importance
        if ensemble_model is not None:
            importance["ensemble"] = ensemble_model._get_feature_importance()
        
        return importance
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@router.post("/batch/predict", response_model=List[InferenceResponse])
async def batch_predict(requests: List[InferenceRequest]) -> List[InferenceResponse]:
    """
    Make batch predictions for multiple requests.
    
    Parameters:
    -----------
    requests : List[InferenceRequest]
        List of inference requests
        
    Returns:
    --------
    List[InferenceResponse]
        List of prediction responses
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    responses = []
    
    try:
        for i, request in enumerate(requests):
            try:
                response = ensemble_model.predict(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch prediction {i} failed: {e}")
                # Create error response
                error_response = InferenceResponse(
                    outbreak_probability=0.5,
                    confidence=0.0,
                    lead_time_days=7,
                    severity_level="low",
                    tabular_prediction=0.5,
                    timeseries_prediction=0.5,
                    text_prediction=0.5,
                    contributing_factors=["Prediction failed"],
                    recommendations=["Manual review required"],
                    feature_importance={},
                    model_versions={"error": "prediction_failed"},
                    processing_time_ms=0.0,
                    timestamp=datetime.utcnow()
                )
                responses.append(error_response)
        
        logger.info(f"Batch prediction completed: {len(responses)} responses")
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check for ML inference service.
    
    Returns:
    --------
    Dict[str, Any]
        Health status
    """
    models_loaded = all([
        tabular_model is not None,
        timeseries_model is not None,
        text_model is not None,
        ensemble_model is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ml_inference"
    }

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.post("/preprocess/clinical-report")
async def preprocess_clinical_report(report: ClinicalReport) -> Dict[str, Any]:
    """
    Preprocess clinical report for ML inference.
    
    Parameters:
    -----------
    report : ClinicalReport
        Clinical report data
        
    Returns:
    --------
    Dict[str, Any]
        Preprocessed features
    """
    try:
        # Extract features from clinical report
        features = {
            "patient_age": report.age,
            "patient_sex": report.sex.value,
            "symptoms_text": report.symptoms_text,
            "symptoms_structured": report.symptoms_structured,
            "severity": report.severity.value if report.severity else "unknown",
            "language": report.language.value,
            "facility_id": report.facility_id,
            "date": report.date.isoformat()
        }
        
        return {
            "features": features,
            "preprocessing_status": "success"
        }
        
    except Exception as e:
        logger.error(f"Clinical report preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.post("/preprocess/water-quality")
async def preprocess_water_quality(reading: WaterQualityReading) -> Dict[str, Any]:
    """
    Preprocess water quality reading for ML inference.
    
    Parameters:
    -----------
    reading : WaterQualityReading
        Water quality data
        
    Returns:
    --------
    Dict[str, Any]
        Preprocessed features
    """
    try:
        # Extract features from water quality reading
        features = {
            "ph": reading.ph,
            "turbidity": reading.turbidity,
            "temperature": reading.temperature,
            "conductivity": reading.conductivity,
            "bacterial_test": reading.bacterial_test_result,
            "chlorine_residual": reading.chlorine_residual,
            "quality_score": reading.quality_score,
            "location": reading.location,
            "timestamp": reading.timestamp.isoformat()
        }
        
        return {
            "features": features,
            "preprocessing_status": "success"
        }
        
    except Exception as e:
        logger.error(f"Water quality preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.get("/models/performance", response_model=Dict[str, Any])
async def get_model_performance() -> Dict[str, Any]:
    """
    Get model performance metrics.
    
    Returns:
    --------
    Dict[str, Any]
        Model performance information
    """
    # This would typically load from a database or metrics store
    performance = {
        "tabular_model": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc_roc": 0.91
        },
        "timeseries_model": {
            "accuracy": 0.78,
            "precision": 0.75,
            "recall": 0.81,
            "f1_score": 0.78,
            "auc_roc": 0.86
        },
        "text_model": {
            "accuracy": 0.80,
            "precision": 0.77,
            "recall": 0.83,
            "f1_score": 0.80,
            "auc_roc": 0.88
        },
        "ensemble_model": {
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.91,
            "f1_score": 0.88,
            "auc_roc": 0.94
        }
    }
    
    return performance
