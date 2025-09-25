from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import json
from typing import List, Dict, Any
from datetime import datetime
import asyncio
import aiohttp
import os
from config.firebase import get_firestore_client

router = APIRouter()

# Mock data for demonstration
MOCK_OFFICIALS = [
    {"name": "Dr. Sarah Johnson", "phone": "+91-9876543210", "role": "District Health Officer"},
    {"name": "Mr. Rajesh Kumar", "phone": "+91-9876543211", "role": "Block Development Officer"},
    {"name": "Ms. Priya Sharma", "phone": "+91-9876543212", "role": "ASHA Supervisor"}
]

MOCK_CITIZENS = [
    {"name": "Village A Residents", "phone": "+91-9876543213", "location": "Village A"},
    {"name": "Village B Residents", "phone": "+91-9876543214", "location": "Village B"},
    {"name": "Village C Residents", "phone": "+91-9876543215", "location": "Village C"}
]

@router.post("/emergency")
async def send_emergency_alert(alert_data: Dict[str, Any]):
    """
    Send emergency alerts to officials and citizens
    """
    try:
        # Process alert data
        alert_type = alert_data.get("type", "general")
        severity = alert_data.get("severity", "medium")
        location = alert_data.get("location", "Unknown")
        message = alert_data.get("message", "Emergency alert")
        coordinates = alert_data.get("coordinates", {})
        reported_by = alert_data.get("reportedBy", "System")
        
        # Create alert record
        alert_record = {
            "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": alert_type,
            "severity": severity,
            "location": location,
            "message": message,
            "coordinates": coordinates,
            "reportedBy": reported_by,
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }
        
        # Store in Firestore
        db = get_firestore_client()
        db.collection("emergency_alerts").add(alert_record)
        
        # Send notifications based on severity
        notifications_sent = []
        
        if severity in ["high", "critical"]:
            # Send to all officials
            for official in MOCK_OFFICIALS:
                notification = await send_notification(
                    official["phone"], 
                    f"🚨 EMERGENCY ALERT 🚨\n\n{message}\n\nLocation: {location}\nReported by: {reported_by}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "emergency"
                )
                notifications_sent.append({
                    "recipient": official["name"],
                    "phone": official["phone"],
                    "status": notification["status"]
                })
            
            # Send to citizens in affected area
            for citizen in MOCK_CITIZENS:
                if location.lower() in citizen["location"].lower():
                    notification = await send_notification(
                        citizen["phone"],
                        f"⚠️ HEALTH ALERT ⚠️\n\n{message}\n\nPlease take necessary precautions and contact health authorities if needed.\n\nLocation: {location}",
                        "citizen_alert"
                    )
                    notifications_sent.append({
                        "recipient": citizen["name"],
                        "phone": citizen["phone"],
                        "status": notification["status"]
                    })
        
        return {
            "success": True,
            "alert_id": alert_record["id"],
            "notifications_sent": len(notifications_sent),
            "details": notifications_sent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending emergency alert: {str(e)}")

@router.post("/multilingual")
async def send_multilingual_alert(alert_data: Dict[str, Any]):
    """
    Send multilingual emergency alerts
    """
    try:
        message = alert_data.get("message", "Emergency alert")
        location = alert_data.get("location", "Unknown")
        languages = alert_data.get("languages", ["en", "hi", "te"])
        
        # Translation mapping (in real implementation, use Translate API)
        translations = {
            "en": message,
            "hi": f"आपातकालीन चेतावनी! {message} स्थान: {location}",
            "bn": f"জরুরি সতর্কতা! {message} অবস্থান: {location}",
            "as": f"জৰুৰী সতৰ্কবাৰ্তা! {message} স্থান: {location}",
            "ne": f"आपतकालीन सूचना! {message} स्थान: {location}",
            "brx": f"खथाय हरनाय! {message} थाम: {location}",
            "lus": f"Thlamuang zawhna! {message} Hmun: {location}",
            "mni": f"ꯑꯣꯞꯇꯦꯛ ꯂꯥꯔꯠ! {message} ꯃꯁꯛ: {location}",
            "kha": f"Ka jingmaham kyrkieh! {message} jaka: {location}",
            "grt": f"Nokgijagipa dakgrikani! {message} chiktang: {location}"
        }
        
        notifications_sent = []
        
        for lang in languages:
            translated_message = translations.get(lang, message)
            
            # Send to officials
            for official in MOCK_OFFICIALS:
                notification = await send_notification(
                    official["phone"],
                    translated_message,
                    "multilingual_emergency"
                )
                notifications_sent.append({
                    "recipient": official["name"],
                    "language": lang,
                    "status": notification["status"]
                })
        
        return {
            "success": True,
            "languages_sent": languages,
            "notifications_sent": len(notifications_sent)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending multilingual alert: {str(e)}")

async def send_notification(phone: str, message: str, notification_type: str):
    """
    Mock function to send SMS/call notifications
    In real implementation, integrate with SMS gateway and calling service
    """
    try:
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Mock response - in real implementation, call actual SMS/calling API
        return {
            "status": "sent",
            "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "phone": phone,
            "type": notification_type
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "phone": phone,
            "type": notification_type
        }

@router.post("/water/upload")
async def upload_water_data(
    file: UploadFile = File(...),
    type: str = "water_quality",
    uploadedBy: str = "Unknown"
):
    """
    Upload and process water quality CSV data
    """
    try:
        # Read file content
        content = await file.read()
        
        # Parse CSV
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Process data and generate alerts
        processed_count = 0
        alerts_generated = 0
        
        for _, row in df.iterrows():
            # Extract water quality data
            ph = row.get('ph', row.get('pH', 7.0))
            turbidity = row.get('turbidity', row.get('Turbidity', 1.0))
            location = row.get('location', row.get('Location', 'Unknown'))
            
            # Determine status
            if ph < 6.5 or ph > 8.5 or turbidity > 5:
                status = "unsafe"
                alerts_generated += 1
            elif turbidity > 3:
                status = "caution"
            else:
                status = "safe"
            
            # Store in database
            water_record = {
                "location": location,
                "ph": float(ph),
                "turbidity": float(turbidity),
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "uploadedBy": uploadedBy,
                "source": "csv_upload"
            }
            
            # Store in Firestore
            db = get_firestore_client()
            db.collection("water_quality").add(water_record)
            
            processed_count += 1
            
            # Send alert if unsafe
            if status == "unsafe":
                await send_emergency_alert({
                    "type": "water_quality_emergency",
                    "severity": "high",
                    "location": location,
                    "message": f"URGENT: Unsafe water quality detected at {location}. pH: {ph}, Turbidity: {turbidity} NTU",
                    "coordinates": {"lat": 17.3850, "lng": 78.4867},
                    "reportedBy": uploadedBy
                })
        
        return {
            "success": True,
            "processedCount": processed_count,
            "alertsGenerated": alerts_generated,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/health/upload")
async def upload_health_data(
    file: UploadFile = File(...),
    type: str = "health_reports",
    uploadedBy: str = "Unknown"
):
    """
    Upload and process health reports CSV data
    """
    try:
        # Read file content
        content = await file.read()
        
        # Parse CSV
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Process data and generate alerts
        processed_count = 0
        alerts_generated = 0
        
        for _, row in df.iterrows():
            # Extract health data
            patient_name = row.get('patient_name', row.get('Patient Name', 'Unknown'))
            symptoms = str(row.get('symptoms', row.get('Symptoms', ''))).split(',')
            location = row.get('location', row.get('Location', 'Unknown'))
            severity = row.get('severity', row.get('Severity', 'low'))
            
            # Determine if alert is needed
            critical_symptoms = ['fever', 'diarrhea', 'vomiting', 'chest pain', 'difficulty breathing']
            has_critical_symptoms = any(symptom.lower().strip() in critical_symptoms for symptom in symptoms)
            
            if severity in ['high', 'critical'] or has_critical_symptoms:
                alerts_generated += 1
                
                # Send emergency alert
                await send_emergency_alert({
                    "type": "health_emergency",
                    "severity": severity,
                    "location": location,
                    "message": f"URGENT: {severity.upper()} health alert - {patient_name} showing symptoms: {', '.join(symptoms)}. Location: {location}",
                    "coordinates": {"lat": 17.3850, "lng": 78.4867},
                    "reportedBy": uploadedBy
                })
            
            # Store in database
            health_record = {
                "patientName": patient_name,
                "symptoms": [s.strip() for s in symptoms if s.strip()],
                "location": location,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "uploadedBy": uploadedBy,
                "source": "csv_upload"
            }
            
            # Store in Firestore
            db = get_firestore_client()
            db.collection("health_reports").add(health_record)
            
            processed_count += 1
        
        return {
            "success": True,
            "processedCount": processed_count,
            "alertsGenerated": alerts_generated,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/alerts/history")
async def get_alert_history():
    """
    Get emergency alert history
    """
    try:
        db = get_firestore_client()
        alerts = db.collection("emergency_alerts").order_by("timestamp", direction="DESCENDING").limit(50).stream()
        
        alert_list = []
        for alert in alerts:
            alert_data = alert.to_dict()
            alert_data["id"] = alert.id
            alert_list.append(alert_data)
        
        return {
            "success": True,
            "alerts": alert_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alert history: {str(e)}")
