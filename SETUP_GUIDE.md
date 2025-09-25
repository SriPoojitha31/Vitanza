# Vitanza - Smart Health Surveillance System Setup Guide

## Overview
This is a comprehensive Smart Health Surveillance and Early Warning System designed to detect, monitor, and help prevent outbreaks of water-borne diseases in vulnerable communities. The system includes AI/ML models, real-time alerts, multilingual support, and role-based access control.

## Features Implemented

### ✅ Core Features
- **Firebase Authentication** - Secure user authentication with role-based access
- **Real-time Dashboard** - Comprehensive health and water quality monitoring
- **GIS Mapping** - Interactive maps with Leaflet for health data visualization
- **Water Quality Monitoring** - Advanced water quality reporting with CSV upload
- **Health Reports** - Symptom tracking and outbreak detection
- **Emergency Alerts** - Automatic SMS/call notifications for critical conditions
- **Multilingual Support** - Alerts in English, Hindi, Telugu, Bengali, Tamil, Gujarati
- **File Upload** - CSV/Excel file processing for bulk data import
- **Role-based Access** - Different access levels for Government, ASHA workers, volunteers

### 🔧 Technical Features
- **Frontend**: React 18 with Vite, Firebase Auth, Leaflet maps, Lucide icons
- **Backend**: FastAPI with Firebase Admin SDK, Pandas for data processing
- **Database**: Firebase Firestore for real-time data storage
- **Maps**: Interactive GIS mapping with OpenStreetMap
- **Alerts**: Mock SMS/call system (ready for integration with actual services)

## Installation & Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Firebase project with Authentication and Firestore enabled
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Vitanza-main
```

### 2. Backend Setup

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Firebase Configuration
1. Create a Firebase project at https://console.firebase.google.com
2. Enable Authentication and Firestore Database
3. Generate a service account key:
   - Go to Project Settings > Service Accounts
   - Click "Generate new private key"
   - Save the JSON file as `firebase-service-account.json` in the backend directory

#### Environment Variables
Create a `.env` file in the backend directory:
```env
FIREBASE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"your-project-id",...}
JWT_SECRET=your-jwt-secret-key
```

#### Run Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

#### Install Dependencies
```bash
cd frontend
npm install
```

#### Firebase Configuration
Create a `.env` file in the frontend directory:
```env
VITE_FIREBASE_API_KEY=your-api-key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=your-app-id
VITE_API_BASE=http://localhost:8000
```

#### Run Frontend
```bash
cd frontend
npm run dev
```

## User Roles & Permissions

### Admin
- Full system access
- User management
- All reporting features
- Emergency alert management
- Analytics and monitoring

### Officer (Government/Health Authority)
- View and manage reports
- Upload CSV files
- Send emergency alerts
- Access analytics
- Manage communities

### Worker (ASHA/Health Worker)
- View and add reports
- Upload CSV files
- Manage water quality and health reports
- View community data

### Community
- View reports
- Add basic reports
- View community information

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration

### Water Quality
- `GET /api/water` - Get water quality reports
- `POST /api/water` - Add water quality report
- `POST /api/water/upload` - Upload CSV file

### Health Reports
- `GET /api/health` - Get health reports
- `POST /api/health` - Add health report
- `POST /api/health/upload` - Upload CSV file

### Emergency Alerts
- `POST /api/emergency` - Send emergency alert
- `POST /api/multilingual` - Send multilingual alert
- `GET /api/alerts/history` - Get alert history

### GIS & Mapping
- `GET /api/gis/heatmap` - Get heatmap data
- `GET /api/alerts` - Get alert data for mapping

## File Upload Format

### Water Quality CSV Format
```csv
location,ph,turbidity,temperature,chlorine,bacteria
"Village A - Main Well",7.2,0.5,22,0.8,negative
"Village B - Water Tank",6.8,2.1,25,0.3,positive
```

### Health Reports CSV Format
```csv
patient_name,age,gender,symptoms,location,severity,temperature,blood_pressure,contact_number
"Rajesh Kumar",35,Male,"Fever,Diarrhea,Vomiting","Village A",high,102.5,"140/90","+91-9876543210"
"Sunita Devi",28,Female,"Cough,Fatigue","Village B",medium,99.2,"120/80","+91-9876543211"
```

## Emergency Alert System

### Automatic Triggers
- **Water Quality**: pH < 6.5 or > 8.5, Turbidity > 5 NTU
- **Health Reports**: Critical symptoms, High/Critical severity
- **Outbreak Detection**: Multiple similar symptoms in same area

### Alert Channels
- SMS notifications to officials and citizens
- Multilingual support (6 Indian languages)
- Real-time dashboard updates
- GIS map markers

## Multilingual Support

The system supports alerts in:
- English (en)
- Hindi (hi) - हिंदी
- Telugu (te) - తెలుగు
- Bengali (bn) - বাংলা
- Tamil (ta) - தமிழ்
- Gujarati (gu) - ગુજરાતી

## GIS Mapping Features

- Interactive maps with OpenStreetMap
- Layer toggles for different data types
- Custom markers for alerts, water quality, communities
- Real-time data visualization
- Emergency alert sending from map interface

## Development & Testing

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

### Linting
```bash
# Backend
cd backend
flake8 .

# Frontend
cd frontend
npm run lint
```

## Production Deployment

### Backend (FastAPI)
- Use Gunicorn or similar WSGI server
- Set up proper environment variables
- Configure Firebase service account
- Set up database backups

### Frontend (React)
- Build for production: `npm run build`
- Serve with Nginx or similar
- Configure environment variables
- Set up CDN for static assets

### Firebase
- Enable Firestore security rules
- Set up proper authentication rules
- Configure storage buckets
- Set up monitoring and alerts

## Troubleshooting

### Common Issues

1. **Firebase Authentication Errors**
   - Check API keys in environment variables
   - Verify Firebase project configuration
   - Ensure Authentication is enabled

2. **File Upload Issues**
   - Check file format (CSV/Excel)
   - Verify file size limits
   - Check backend logs for errors

3. **Map Not Loading**
   - Check internet connection
   - Verify Leaflet CSS is loaded
   - Check browser console for errors

4. **API Connection Issues**
   - Verify backend is running on correct port
   - Check CORS configuration
   - Verify API_BASE environment variable

## Security Considerations

- All API endpoints require authentication
- Role-based access control implemented
- Input validation and sanitization
- Secure file upload handling
- Firebase security rules for data access

## Monitoring & Analytics

- Real-time dashboard with key metrics
- Alert history and tracking
- User activity monitoring
- System health checks
- Performance metrics

## Support & Maintenance

- Regular security updates
- Database maintenance
- Performance optimization
- Feature enhancements
- Bug fixes and patches

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For support or questions, please contact the development team or create an issue in the repository.
