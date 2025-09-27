# Vitanza ML Architecture Guide

## Overview

This document describes the comprehensive machine learning architecture implemented for the Vitanza outbreak prediction system. The system uses an ensemble approach combining multiple specialized models to predict disease outbreaks with high accuracy and multilingual support.

## Architecture Overview

```
[Mobile App / SMS / IoT sensors] → FastAPI ingestion endpoints → Message queue (optional) → Preprocessing → Models
                                                                                        \
                                                                                         → Database (Postgres / Timescale / Mongo) → Dashboard
```

## Data Collection & Models

### 1. Clinical Reports
- **Patient Data**: `patient_id*`, age, sex, symptoms (text + structured), date, facility_id
- **Privacy**: Uses hashed IDs for patient privacy
- **Multilingual**: Supports Bengali, Hindi, Telugu, Assamese, English

### 2. Environmental Data
- **Water Quality**: pH, turbidity, temperature, conductivity, bacterial test results
- **Weather**: Rainfall, season, temperature, humidity
- **Location**: Village/ward, lat/lon coordinates

### 3. Contextual Data
- **Population**: Density, age distribution, sanitation index
- **Events**: Festival periods, natural disasters, health campaigns
- **Geographic**: Village/ward, district, state, altitude

## ML Model Architecture

### 1. Tabular Model (XGBoost)
**Purpose**: Analyze aggregated features and demographic data

**Features**:
- 7-day, 14-day, 30-day rolling averages of case counts
- Water quality metrics (pH, turbidity, temperature)
- Population density and sanitation indices
- Geographic features (latitude, longitude, altitude)
- Event flags (festivals, disasters, outbreaks)

**Implementation**: `backend/ml/tabular_model.py`
```python
# Example usage
model = TabularOutbreakPredictor("models/xgb_outbreak.json")
features = TabularFeatures(
    cases_7d_avg=5.0,
    water_quality_7d_avg=7.0,
    population_density=500.0,
    # ... other features
)
prediction = model.predict(features)
```

### 2. Time-Series Model (LSTM)
**Purpose**: Analyze temporal patterns in sensor data and case counts

**Features**:
- 30-day sliding windows of sensor readings
- Case count sequences
- Environmental factor trends
- Seasonal patterns

**Implementation**: `backend/ml/timeseries_model.py`
```python
# Example usage
model = TimeSeriesOutbreakPredictor("models/ts_lstm.pth")
window = TimeSeriesWindow(
    timestamps=[...],
    sensor_readings=[...],
    case_counts=[...],
    environmental_factors=[...],
    window_size=30
)
prediction = model.predict(window)
```

### 3. Text Model (XLM-Roberta)
**Purpose**: Analyze clinical text in multiple languages

**Features**:
- Multilingual symptom descriptions
- Severity indicators
- Keyword extraction
- Sentiment analysis

**Implementation**: `backend/ml/text_model.py`
```python
# Example usage
model = MultilingualTextClassifier("models/text_xlm_roberta")
text_features = TextFeatures(
    text="রোগীর জ্বর এবং ডায়রিয়া",
    language=LanguageCode.BENGALI
)
prediction = model.predict(text_features)
```

### 4. Ensemble Meta-Learner
**Purpose**: Combine predictions from all models for final outbreak probability

**Features**:
- Individual model predictions
- Model confidence scores
- Prediction agreement metrics
- Meta-features from data quality

**Implementation**: `backend/ml/ensemble_model.py`
```python
# Example usage
ensemble = EnsembleOutbreakPredictor("models/ensemble_meta.joblib")
ensemble.set_base_models(tabular_model, timeseries_model, text_model)
response = ensemble.predict(inference_request)
```

## Data Preprocessing Pipeline

### 1. Clinical Data Preprocessing
- **Text Processing**: Language detection, symptom keyword extraction
- **Feature Engineering**: Age groups, symptom categories, severity indicators
- **Missing Value Handling**: Median imputation for numeric, "unknown" for categorical

### 2. Environmental Data Preprocessing
- **Sensor Data**: Outlier detection using IQR method
- **Quality Index**: Calculated from pH, turbidity, temperature
- **Contamination Risk**: Based on bacterial tests and chlorine levels

### 3. Contextual Data Preprocessing
- **Temporal Features**: Season, day of week, monsoon periods
- **Interaction Features**: Population density × sanitation index
- **Geographic Features**: Distance calculations, altitude effects

## Multilingual Support

### 1. Language Detection
```python
# Automatic language detection
detected_lang = translator.detect_language("রোগীর জ্বর এবং ডায়রিয়া")
# Returns: "bn" (Bengali)
```

### 2. Translation Fallback System
- **Primary**: Hugging Face translation models
- **Secondary**: Google Translate API
- **Fallback**: Pattern-based keyword translation

### 3. Supported Languages
- English (en)
- Bengali (bn)
- Hindi (hi)
- Telugu (te)
- Assamese (as)

## API Endpoints

### 1. Main Prediction Endpoint
```http
POST /api/ml/predict
Content-Type: application/json

{
  "device_id": "sensor_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "features_tabular": { ... },
  "timeseries_window": { ... },
  "clinical_text": "Patient has fever and diarrhea",
  "text_language": "en",
  "location": { ... },
  "environmental": { ... },
  "population": { ... },
  "sanitation": { ... },
  "events": { ... }
}
```

### 2. Individual Model Endpoints
```http
POST /api/ml/predict/tabular
POST /api/ml/predict/timeseries
POST /api/ml/predict/text
```

### 3. Model Management Endpoints
```http
GET /api/ml/models/status
POST /api/ml/models/reload
GET /api/ml/models/feature-importance
```

## Training Pipeline

### 1. Data Preparation
```bash
# Create sample data
python backend/scripts/train_models.py --data_path data --output_dir models
```

### 2. Model Training
```python
# Train all models
trainer = ModelTrainer(output_dir="models")
results = trainer.train_all_models("data/")
```

### 3. Model Evaluation
- Cross-validation with 5-fold stratified splits
- AUC-ROC and AUC-PR metrics
- Feature importance analysis
- Model performance tracking

## Deployment Architecture

### 1. Model Loading
- Models loaded on FastAPI startup
- Automatic fallback if models unavailable
- Health checks for model status

### 2. Inference Pipeline
1. **Data Validation**: Pydantic schema validation
2. **Preprocessing**: Feature engineering and scaling
3. **Individual Predictions**: Tabular, time-series, text models
4. **Ensemble Prediction**: Meta-learner combination
5. **Response Generation**: Confidence scores, explanations, recommendations

### 3. Performance Optimization
- **Batch Processing**: Multiple predictions in single request
- **Caching**: Model predictions for repeated queries
- **Async Processing**: Non-blocking inference
- **Resource Management**: GPU/CPU optimization

## Model Performance

### Expected Performance Metrics
- **Tabular Model**: AUC-ROC ~0.91, Precision ~0.82, Recall ~0.88
- **Time-Series Model**: AUC-ROC ~0.86, handles temporal patterns
- **Text Model**: AUC-ROC ~0.88, multilingual support
- **Ensemble Model**: AUC-ROC ~0.94, best overall performance

### Model Interpretability
- **SHAP Values**: Feature importance for tabular model
- **Attention Weights**: Important timesteps for LSTM
- **Keyword Analysis**: Critical symptoms for text model
- **Ensemble Explanations**: Contributing factors and recommendations

## Data Quality & Monitoring

### 1. Data Validation
- Schema validation for all input data
- Range checks for sensor readings
- Language detection accuracy
- Missing value detection

### 2. Model Monitoring
- Prediction confidence tracking
- Model performance drift detection
- Feature importance monitoring
- A/B testing capabilities

### 3. Alert System
- High-risk predictions trigger alerts
- Model performance degradation warnings
- Data quality issues notifications
- System health monitoring

## Security & Privacy

### 1. Data Privacy
- Patient ID hashing for privacy
- Data encryption in transit and at rest
- Access control and authentication
- Audit logging for all operations

### 2. Model Security
- Model versioning and integrity checks
- Secure model storage and loading
- Input validation and sanitization
- Rate limiting and abuse prevention

## Deployment & Scaling

### 1. Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Production Deployment
- Docker containerization
- Kubernetes orchestration
- Load balancing and auto-scaling
- Health checks and monitoring

### 3. Model Updates
- Blue-green deployment for model updates
- A/B testing for new models
- Rollback capabilities
- Performance monitoring

## Usage Examples

### 1. Basic Prediction
```python
import requests

# Make prediction request
response = requests.post("http://localhost:8000/api/ml/predict", json={
    "device_id": "sensor_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "features_tabular": {
        "cases_7d_avg": 5.0,
        "water_quality_7d_avg": 7.0,
        "population_density": 500.0,
        # ... other features
    },
    "timeseries_window": {
        "timestamps": [...],
        "sensor_readings": [...],
        "case_counts": [...],
        "environmental_factors": [...],
        "window_size": 30
    },
    "clinical_text": "Patient has severe fever and diarrhea",
    "text_language": "en",
    # ... other data
})

result = response.json()
print(f"Outbreak probability: {result['outbreak_probability']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Lead time: {result['lead_time_days']} days")
```

### 2. Batch Prediction
```python
# Multiple predictions in single request
batch_requests = [request1, request2, request3]
response = requests.post("http://localhost:8000/api/ml/batch/predict", json=batch_requests)
```

### 3. Model Status Check
```python
# Check model status
response = requests.get("http://localhost:8000/api/ml/models/status")
status = response.json()
print(f"Models loaded: {status['tabular_model']['loaded']}")
```

## Troubleshooting

### 1. Common Issues
- **Model Loading Errors**: Check model files exist and are readable
- **Memory Issues**: Reduce batch size or use model quantization
- **Translation Errors**: Check internet connection for Google Translate
- **Performance Issues**: Monitor GPU/CPU usage and optimize accordingly

### 2. Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. Health Checks
```bash
# Check API health
curl http://localhost:8000/api/ml/health

# Check model status
curl http://localhost:8000/api/ml/models/status
```

## Future Enhancements

### 1. Model Improvements
- **Transformer Models**: Replace LSTM with Transformer for time-series
- **Graph Neural Networks**: Model relationships between locations
- **Federated Learning**: Train models across multiple facilities
- **Active Learning**: Improve models with human feedback

### 2. Feature Enhancements
- **Satellite Imagery**: Environmental monitoring from space
- **Social Media**: Public health sentiment analysis
- **Mobile Data**: Population movement patterns
- **Weather Data**: Advanced meteorological features

### 3. System Improvements
- **Real-time Streaming**: Apache Kafka for real-time data
- **Model Serving**: TensorFlow Serving or TorchServe
- **Monitoring**: Prometheus and Grafana dashboards
- **CI/CD**: Automated model training and deployment

---

This comprehensive ML architecture provides a robust, scalable, and accurate system for outbreak prediction with multilingual support and real-time inference capabilities.
