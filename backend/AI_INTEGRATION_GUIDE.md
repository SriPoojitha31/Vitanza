# Vitanza AI Integration Guide

## Overview

This guide explains the comprehensive AI integration implemented in the Vitanza health dashboard system, including LangChain with LLM capabilities, multilingual support, and frontend dashboard updates.

## 🤖 AI Assistant Features

### 1. LangChain Integration
- **Model**: BLOOM-560m (multilingual, community health focused)
- **Framework**: LangChain with HuggingFace Pipeline
- **Capabilities**: Symptom analysis, outbreak explanations, health awareness

### 2. Multilingual Support
- **Languages**: English, Bengali, Hindi, Telugu, Assamese
- **Features**: Automatic language detection, translation fallback
- **Use Cases**: Clinical text analysis, community health messages

### 3. AI Tools & Agents
- **Translation Tool**: Auto-detect language, translate for ML processing
- **Database Tool**: Query outbreak history from database
- **IoT Sensor Tool**: Real-time sensor data integration
- **Health Advice Tool**: Symptom-based health recommendations

## 🚀 API Endpoints

### AI Assistant Endpoints

#### 1. Symptom Analysis
```http
POST /api/ai/analyze-symptoms
Content-Type: application/json

{
  "symptoms": "fever, diarrhea, stomach cramps",
  "language": "English",
  "patient_id": "optional",
  "location": "optional"
}
```

**Response:**
```json
{
  "analysis": "Based on your symptoms, this could indicate a water-borne disease...",
  "language": "English",
  "timestamp": "2024-01-15T10:30:00Z",
  "symptoms": "fever, diarrhea, stomach cramps",
  "recommendations": [
    "Seek medical attention if symptoms worsen",
    "Maintain proper hydration",
    "Use only safe, treated water"
  ],
  "severity_level": "medium",
  "water_borne_risk": true
}
```

#### 2. Outbreak Explanation
```http
POST /api/ai/explain-outbreak
Content-Type: application/json

{
  "prediction_data": {
    "outbreak_probability": 0.75,
    "confidence": 0.85,
    "lead_time_days": 7,
    "severity_level": "high",
    "contributing_factors": ["water_contamination", "population_density"]
  },
  "language": "Bengali",
  "community": "Green Valley"
}
```

**Response:**
```json
{
  "explanation": "The AI model predicts a 75% chance of outbreak in the next 7 days...",
  "language": "Bengali",
  "timestamp": "2024-01-15T10:30:00Z",
  "risk_level": "high",
  "preventive_measures": [
    "Increase water quality monitoring",
    "Enhance sanitation measures",
    "Activate emergency protocols"
  ],
  "community_actions": [
    "Inform community leaders",
    "Conduct awareness campaigns",
    "Monitor high-risk individuals"
  ]
}
```

#### 3. Health Awareness Messages
```http
POST /api/ai/create-awareness
Content-Type: application/json

{
  "topic": "water_safety",
  "community": "Riverside Village",
  "language": "Hindi",
  "audience": "general"
}
```

**Response:**
```json
{
  "message": "सुरक्षित पानी के लिए महत्वपूर्ण सुझाव...",
  "topic": "water_safety",
  "community": "Riverside Village",
  "language": "Hindi",
  "audience": "general",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 4. AI Chat Agent
```http
POST /api/ai/chat
Content-Type: application/json

{
  "query": "What should I do if I see contaminated water in my village?",
  "context": {
    "location": "Green Valley",
    "user_role": "health_worker"
  }
}
```

**Response:**
```json
{
  "response": "If you notice contaminated water, immediately...",
  "timestamp": "2024-01-15T10:30:00Z",
  "query": "What should I do if I see contaminated water in my village?",
  "confidence": 0.9,
  "sources": ["Health Knowledge Base", "Database", "IoT Sensors"]
}
```

## 🎨 Frontend Dashboard Updates

### 1. Alerts Management Page
**URL**: `/alerts`

**Features:**
- Real-time alert monitoring
- Severity-based color coding
- Action tracking and status updates
- Community impact assessment
- Recent activity timeline

**Key Components:**
- Alert creation modal
- Status update functionality
- People affected tracking
- Action item management

### 2. Community Management Page
**URL**: `/community`

**Features:**
- Community health worker management
- Population and health score tracking
- Coordinator information
- Recent community activities
- Health worker and volunteer counts

**Key Components:**
- Community cards with health scores
- Add community functionality
- Activity timeline
- Health worker profiles

### 3. Health Reports Page
**URL**: `/reports`

**Features:**
- Health incident report submission
- Report filtering and sorting
- Severity and status tracking
- Patient information management
- Report statistics dashboard

**Key Components:**
- Submit report modal
- Report table with filters
- Status and severity indicators
- Export functionality

## 🔧 Implementation Details

### Backend AI Integration

#### 1. LangChain Setup
```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load multilingual model
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=pipe)
```

#### 2. AI Assistant Class
```python
class HealthAIAssistant:
    def __init__(self, model_name="bigscience/bloom-560m"):
        self.llm = None
        self.chains = {}
        self.tools = []
        self.agent = None
        self._initialize_models()
        self._create_tools()
        self._create_chains()
        self._create_agent()
```

#### 3. Multilingual Chains
```python
# Symptom analysis chain
symptom_template = """
You are a health assistant. 
The user reports: {symptoms}.
Based on this, suggest if it could be a water-borne disease, 
and explain in {language}.
"""

symptom_prompt = PromptTemplate(
    input_variables=["symptoms", "language"],
    template=symptom_template
)

symptom_chain = LLMChain(llm=llm, prompt=symptom_prompt)
```

### Frontend Integration

#### 1. Dashboard Components
```jsx
// Alerts Management
const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [showCreateAlert, setShowCreateAlert] = useState(false);
  
  // Alert management logic
  const handleCreateAlert = () => {
    // Create new alert
  };
  
  const updateAlertStatus = (alertId, newStatus) => {
    // Update alert status
  };
  
  return (
    <div className="p-6 space-y-6">
      {/* Alert management UI */}
    </div>
  );
};
```

#### 2. Report Submission
```jsx
// Health Reports
const HealthReports = () => {
  const [showSubmitForm, setShowSubmitForm] = useState(false);
  const [newReport, setNewReport] = useState({
    patientName: '',
    village: '',
    symptoms: '',
    severity: 'medium'
  });
  
  const handleSubmitReport = () => {
    // Submit new health report
  };
  
  return (
    <div className="p-6 space-y-6">
      {/* Report management UI */}
    </div>
  );
};
```

## 📊 Usage Examples

### 1. Symptom Analysis in Bengali
```javascript
const response = await fetch('/api/ai/analyze-symptoms', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symptoms: 'জ্বর এবং ডায়রিয়া',
    language: 'Bengali'
  })
});

const result = await response.json();
console.log(result.analysis); // Bengali health advice
```

### 2. Outbreak Prediction Explanation
```javascript
const response = await fetch('/api/ai/explain-outbreak', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prediction_data: {
      outbreak_probability: 0.8,
      confidence: 0.9,
      lead_time_days: 5
    },
    language: 'Hindi'
  })
});

const explanation = await response.json();
console.log(explanation.explanation); // Hindi explanation
```

### 3. Health Awareness Message
```javascript
const response = await fetch('/api/ai/create-awareness', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    topic: 'water_safety',
    community: 'Riverside Village',
    language: 'Telugu',
    audience: 'general'
  })
});

const message = await response.json();
console.log(message.message); // Telugu awareness message
```

## 🔍 Monitoring & Debugging

### 1. AI Assistant Status
```http
GET /api/ai/assistant/status
```

**Response:**
```json
{
  "status": "healthy",
  "capabilities": {
    "model_name": "bigscience/bloom-560m",
    "llm_loaded": true,
    "chains_available": ["symptom_analysis", "outbreak_explanation"],
    "tools_available": 4,
    "agent_available": true
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. Supported Languages
```http
GET /api/ai/languages/supported
```

**Response:**
```json
{
  "symptom_analysis": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
  "outbreak_explanation": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
  "health_awareness": ["English", "Bengali", "Hindi", "Telugu", "Assamese"],
  "chat": ["English", "Bengali", "Hindi", "Telugu", "Assamese"]
}
```

### 3. Health Topics
```http
GET /api/ai/topics/health
```

**Response:**
```json
[
  {
    "id": "water_safety",
    "name": "Water Safety",
    "description": "Safe water practices and contamination prevention"
  },
  {
    "id": "hygiene",
    "name": "Personal Hygiene",
    "description": "Hand washing and sanitation practices"
  }
]
```

## 🚀 Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Backend Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Test AI Endpoints
```bash
# Test symptom analysis
curl -X POST "http://localhost:8000/api/ai/analyze-symptoms" \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever, diarrhea", "language": "English"}'

# Test outbreak explanation
curl -X POST "http://localhost:8000/api/ai/explain-outbreak" \
  -H "Content-Type: application/json" \
  -d '{"prediction_data": {"outbreak_probability": 0.7}, "language": "Bengali"}'
```

## 🔧 Configuration

### 1. Model Configuration
```python
# backend/ml/ai_assistant.py
class HealthAIAssistant:
    def __init__(self, model_name="bigscience/bloom-560m", 
                 device="cpu", max_length=200):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
```

### 2. Language Support
```python
# Add new language support
self.language_patterns = {
    "bn": r'[\u0980-\u09FF]',  # Bengali
    "hi": r'[\u0900-\u097F]',  # Hindi
    "te": r'[\u0C00-\u0C7F]',  # Telugu
    "as": r'[\u0980-\u09FF]',  # Assamese
    "ta": r'[\u0B80-\u0BFF]',  # Tamil (new)
}
```

### 3. Frontend Routes
```jsx
// frontend/src/App.jsx
<Route path="/alerts" element={<Alerts />} />
<Route path="/community" element={<Communities />} />
<Route path="/reports" element={<HealthReports />} />
```

## 📈 Performance Optimization

### 1. Model Caching
- Cache loaded models in memory
- Lazy loading for unused features
- Model quantization for faster inference

### 2. Response Caching
- Cache common responses
- Redis integration for distributed caching
- TTL-based cache invalidation

### 3. Frontend Optimization
- Lazy loading of dashboard components
- Virtual scrolling for large datasets
- Debounced search and filtering

## 🔒 Security Considerations

### 1. Input Validation
- Sanitize all user inputs
- Validate language codes
- Limit text length for AI processing

### 2. Rate Limiting
- Implement rate limiting for AI endpoints
- User-based request quotas
- API key authentication for external access

### 3. Data Privacy
- Anonymize patient data in AI processing
- Secure model storage and loading
- Audit logging for AI interactions

## 🎯 Future Enhancements

### 1. Advanced AI Features
- **Voice Input**: Speech-to-text for symptom reporting
- **Image Analysis**: Photo-based symptom assessment
- **Predictive Analytics**: Advanced outbreak prediction models
- **Chatbot Integration**: Real-time health assistance

### 2. Mobile Integration
- **React Native**: Mobile app with AI features
- **Offline Support**: Local AI processing
- **Push Notifications**: AI-generated health alerts

### 3. Community Features
- **Social Health**: Community health challenges
- **Gamification**: Health awareness rewards
- **Peer Support**: Community health worker networks

---

This comprehensive AI integration provides multilingual health assistance, intelligent outbreak prediction explanations, and modern dashboard interfaces for the Vitanza health system. The implementation is production-ready with proper error handling, monitoring, and security considerations.
