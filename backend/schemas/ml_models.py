from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

# ============================================================================
# CORE DATA MODELS
# ============================================================================

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class LanguageCode(str, Enum):
    ENGLISH = "en"
    BENGALI = "bn"
    HINDI = "hi"
    TELUGU = "te"
    ASSAMESE = "as"
    AUTO = "auto"

# ============================================================================
# CLINICAL DATA MODELS
# ============================================================================

class ClinicalReport(BaseModel):
    """Clinical report data model with privacy-preserving hashed IDs."""
    patient_id_hash: str = Field(..., description="Hashed patient ID for privacy")
    age: int = Field(..., ge=0, le=120, description="Patient age")
    sex: Gender = Field(..., description="Patient gender")
    symptoms_text: str = Field(..., description="Free-text symptoms description")
    symptoms_structured: List[str] = Field(default=[], description="Structured symptom codes")
    date: datetime = Field(..., description="Report date")
    facility_id: str = Field(..., description="Healthcare facility identifier")
    language: LanguageCode = Field(default=LanguageCode.AUTO, description="Report language")
    severity: Optional[SeverityLevel] = Field(None, description="Clinical severity assessment")
    diagnosis: Optional[str] = Field(None, description="Clinical diagnosis")
    treatment: Optional[str] = Field(None, description="Treatment provided")
    outcome: Optional[str] = Field(None, description="Patient outcome")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional clinical metadata")

class SymptomCategory(BaseModel):
    """Structured symptom categorization."""
    category: str = Field(..., description="Symptom category (e.g., 'gastrointestinal', 'respiratory')")
    symptoms: List[str] = Field(..., description="List of symptoms in this category")
    severity_score: float = Field(..., ge=0, le=10, description="Severity score 0-10")

# ============================================================================
# ENVIRONMENTAL DATA MODELS
# ============================================================================

class WaterQualityReading(BaseModel):
    """Water quality sensor reading."""
    ph: float = Field(..., ge=0, le=14, description="pH level")
    turbidity: float = Field(..., ge=0, description="Turbidity in NTU")
    temperature: float = Field(..., description="Water temperature in Celsius")
    conductivity: float = Field(..., ge=0, description="Electrical conductivity")
    bacterial_test_result: Optional[bool] = Field(None, description="Bacterial contamination test result")
    chlorine_residual: Optional[float] = Field(None, ge=0, description="Chlorine residual in ppm")
    location: str = Field(..., description="Sampling location")
    timestamp: datetime = Field(..., description="Reading timestamp")
    sensor_id: str = Field(..., description="Sensor device identifier")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Overall water quality score")

class EnvironmentalContext(BaseModel):
    """Environmental context data."""
    rainfall_24h: Optional[float] = Field(None, ge=0, description="24-hour rainfall in mm")
    rainfall_7d: Optional[float] = Field(None, ge=0, description="7-day rainfall in mm")
    season: Optional[str] = Field(None, description="Season (monsoon, summer, winter)")
    temperature_avg: Optional[float] = Field(None, description="Average temperature")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    air_quality_index: Optional[float] = Field(None, ge=0, description="Air quality index")
    flood_risk: Optional[float] = Field(None, ge=0, le=1, description="Flood risk score 0-1")
    drought_index: Optional[float] = Field(None, ge=0, le=1, description="Drought index 0-1")

# ============================================================================
# CONTEXTUAL DATA MODELS
# ============================================================================

class PopulationDensity(BaseModel):
    """Population density information."""
    village_ward: str = Field(..., description="Village or ward name")
    population_count: int = Field(..., ge=0, description="Total population")
    area_km2: float = Field(..., gt=0, description="Area in square kilometers")
    density_per_km2: float = Field(..., ge=0, description="Population density per km²")
    age_distribution: Optional[Dict[str, int]] = Field(None, description="Age group distribution")

class SanitationIndex(BaseModel):
    """Sanitation and hygiene index."""
    toilet_coverage: float = Field(..., ge=0, le=100, description="Toilet coverage percentage")
    waste_management: float = Field(..., ge=0, le=100, description="Waste management score")
    water_access: float = Field(..., ge=0, le=100, description="Water access score")
    hygiene_practices: float = Field(..., ge=0, le=100, description="Hygiene practices score")
    overall_index: float = Field(..., ge=0, le=100, description="Overall sanitation index")

class EventFlags(BaseModel):
    """Event and alert flags."""
    festival_period: bool = Field(default=False, description="During festival/celebration period")
    natural_disaster: bool = Field(default=False, description="Natural disaster occurred")
    disease_outbreak: bool = Field(default=False, description="Known disease outbreak")
    water_contamination: bool = Field(default=False, description="Water contamination alert")
    health_campaign: bool = Field(default=False, description="Health awareness campaign")
    emergency_response: bool = Field(default=False, description="Emergency response active")

# ============================================================================
# GEOGRAPHIC DATA MODELS
# ============================================================================

class GeographicLocation(BaseModel):
    """Geographic location data."""
    village_ward: str = Field(..., description="Village or ward name")
    district: str = Field(..., description="District name")
    state: str = Field(..., description="State name")
    country: str = Field(default="India", description="Country name")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    geo_hash: Optional[str] = Field(None, description="Geographic hash for privacy")

# ============================================================================
# LABELING AND OUTBREAK MODELS
# ============================================================================

class OutbreakWindow(BaseModel):
    """Outbreak detection window."""
    start_date: datetime = Field(..., description="Window start date")
    end_date: datetime = Field(..., description="Window end date")
    case_count: int = Field(..., ge=0, description="Number of cases in window")
    population_normalized: float = Field(..., ge=0, description="Cases per 1000 population")
    outbreak_flag: bool = Field(..., description="Whether this window represents an outbreak")
    severity_level: SeverityLevel = Field(..., description="Outbreak severity level")
    threshold_exceeded: bool = Field(..., description="Whether threshold was exceeded")
    expert_verified: bool = Field(default=False, description="Expert verification status")

class OutbreakPrediction(BaseModel):
    """Outbreak prediction result."""
    probability: float = Field(..., ge=0, le=1, description="Outbreak probability 0-1")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence 0-1")
    lead_time_days: int = Field(..., description="Predicted lead time in days")
    severity_prediction: SeverityLevel = Field(..., description="Predicted severity")
    contributing_factors: List[str] = Field(..., description="Key contributing factors")
    recommendations: List[str] = Field(..., description="Prevention recommendations")
    model_components: Dict[str, float] = Field(..., description="Individual model predictions")

# ============================================================================
# FEATURE ENGINEERING MODELS
# ============================================================================

class TabularFeatures(BaseModel):
    """Aggregated tabular features for XGBoost model."""
    # Temporal features
    cases_7d_avg: float = Field(..., description="7-day average case count")
    cases_14d_avg: float = Field(..., description="14-day average case count")
    cases_30d_avg: float = Field(..., description="30-day average case count")
    case_trend: float = Field(..., description="Case trend (slope)")
    
    # Environmental features
    water_quality_7d_avg: float = Field(..., description="7-day average water quality")
    rainfall_7d_total: float = Field(..., description="7-day total rainfall")
    temperature_7d_avg: float = Field(..., description="7-day average temperature")
    
    # Population features
    population_density: float = Field(..., description="Population density")
    sanitation_index: float = Field(..., description="Sanitation index")
    age_median: float = Field(..., description="Median age")
    
    # Event features
    festival_flag: bool = Field(..., description="Festival period flag")
    disaster_flag: bool = Field(..., description="Natural disaster flag")
    
    # Geographic features
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    altitude: float = Field(..., description="Altitude")

class TimeSeriesWindow(BaseModel):
    """Time series window for LSTM model."""
    timestamps: List[datetime] = Field(..., description="Timestamp sequence")
    sensor_readings: List[List[float]] = Field(..., description="Sensor readings per timestep")
    case_counts: List[int] = Field(..., description="Case counts per timestep")
    environmental_factors: List[List[float]] = Field(..., description="Environmental factors per timestep")
    window_size: int = Field(..., description="Window size in timesteps")

class TextFeatures(BaseModel):
    """Text features for multilingual model."""
    text: str = Field(..., description="Clinical text")
    language: LanguageCode = Field(..., description="Detected or specified language")
    translated_text: Optional[str] = Field(None, description="Translated text if needed")
    symptom_keywords: List[str] = Field(default=[], description="Extracted symptom keywords")
    severity_indicators: List[str] = Field(default=[], description="Severity indicators")
    sentiment_score: Optional[float] = Field(None, description="Text sentiment score")

# ============================================================================
# MODEL PREDICTION MODELS
# ============================================================================

class ModelPrediction(BaseModel):
    """Individual model prediction."""
    model_name: str = Field(..., description="Model identifier")
    prediction: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    features_used: List[str] = Field(..., description="Features used by model")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

class EnsemblePrediction(BaseModel):
    """Ensemble prediction result."""
    final_probability: float = Field(..., ge=0, le=1, description="Final ensemble probability")
    individual_predictions: List[ModelPrediction] = Field(..., description="Individual model predictions")
    meta_features: List[float] = Field(..., description="Meta-learner features")
    explanation: Dict[str, Any] = Field(..., description="Prediction explanation")

# ============================================================================
# INFERENCE REQUEST MODELS
# ============================================================================

class InferenceRequest(BaseModel):
    """Request for outbreak prediction inference."""
    device_id: str = Field(..., description="Device identifier")
    timestamp: datetime = Field(..., description="Request timestamp")
    
    # Tabular features
    features_tabular: TabularFeatures = Field(..., description="Aggregated tabular features")
    
    # Time series data
    timeseries_window: TimeSeriesWindow = Field(..., description="Time series window")
    
    # Text data
    clinical_text: str = Field(..., description="Clinical text description")
    text_language: LanguageCode = Field(default=LanguageCode.AUTO, description="Text language")
    
    # Geographic context
    location: GeographicLocation = Field(..., description="Geographic location")
    
    # Environmental context
    environmental: EnvironmentalContext = Field(..., description="Environmental context")
    
    # Population context
    population: PopulationDensity = Field(..., description="Population density data")
    
    # Sanitation context
    sanitation: SanitationIndex = Field(..., description="Sanitation index")
    
    # Event flags
    events: EventFlags = Field(..., description="Event flags")

class InferenceResponse(BaseModel):
    """Response from outbreak prediction inference."""
    outbreak_probability: float = Field(..., ge=0, le=1, description="Outbreak probability")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    lead_time_days: int = Field(..., description="Predicted lead time")
    severity_level: SeverityLevel = Field(..., description="Predicted severity")
    
    # Model components
    tabular_prediction: float = Field(..., description="Tabular model prediction")
    timeseries_prediction: float = Field(..., description="Time series model prediction")
    text_prediction: float = Field(..., description="Text model prediction")
    
    # Explanations
    contributing_factors: List[str] = Field(..., description="Key contributing factors")
    recommendations: List[str] = Field(..., description="Prevention recommendations")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance")
    
    # Metadata
    model_versions: Dict[str, str] = Field(..., description="Model versions used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Response timestamp")

# ============================================================================
# TRAINING AND EVALUATION MODELS
# ============================================================================

class TrainingData(BaseModel):
    """Training data structure."""
    features: Dict[str, Any] = Field(..., description="Feature data")
    labels: List[bool] = Field(..., description="Outbreak labels")
    metadata: Dict[str, Any] = Field(..., description="Training metadata")
    data_sources: List[str] = Field(..., description="Data source identifiers")

class ModelEvaluation(BaseModel):
    """Model evaluation metrics."""
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    auc_roc: float = Field(..., description="AUC-ROC score")
    auc_pr: float = Field(..., description="AUC-PR score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance")

class ModelPerformance(BaseModel):
    """Model performance tracking."""
    model_name: str = Field(..., description="Model identifier")
    evaluation: ModelEvaluation = Field(..., description="Evaluation metrics")
    training_date: datetime = Field(..., description="Training date")
    data_size: int = Field(..., description="Training data size")
    performance_trend: List[float] = Field(..., description="Performance over time")
