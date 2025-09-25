from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime, timedelta
import secrets
from mongo import get_mongo_db
from routes.auth_mongo import require_roles

router = APIRouter()

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "vitanza.buzz@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

class EmailVerification(BaseModel):
    email: EmailStr
    token: str
    expires_at: datetime

class NotificationRequest(BaseModel):
    title: str
    message: str
    user_id: Optional[str] = None
    severity: str = "info"

def send_email(to_email: str, subject: str, html_content: str):
    """Send email using Gmail SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

@router.post("/send-verification")
async def send_verification_email(email: EmailStr, db=Depends(get_mongo_db)):
    """Send verification email to user"""
    # Generate verification token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)
    
    # Store verification token in database
    await db.email_verifications.insert_one({
        "email": email,
        "token": token,
        "expires_at": expires_at,
        "created_at": datetime.utcnow()
    })
    
    # Create beautiful email template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome to Vitanza</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
            .logo {{ font-size: 32px; font-weight: bold; margin-bottom: 10px; }}
            .content {{ padding: 40px 30px; }}
            .button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 20px 0; }}
            .footer {{ background: #f8fafc; padding: 30px; text-align: center; color: #64748b; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🌊 Vitanza</div>
                <p style="margin: 0; opacity: 0.9;">Smart Health Surveillance System</p>
            </div>
            <div class="content">
                <h2 style="color: #1e293b; margin-top: 0;">Welcome to Vitanza!</h2>
                <p style="color: #475569; line-height: 1.6;">Thank you for joining our community health monitoring platform. Please verify your email address to complete your registration.</p>
                <div style="text-align: center;">
                    <a href="https://vitanza.buzz/verify?token={token}" class="button">Verify Email Address</a>
                </div>
                <p style="color: #64748b; font-size: 14px; margin-top: 30px;">This verification link will expire in 24 hours.</p>
                <p style="color: #64748b; font-size: 14px;">If you didn't create an account with Vitanza, please ignore this email.</p>
            </div>
            <div class="footer">
                <p>© 2024 Vitanza. All rights reserved.</p>
                <p>Smart Health Surveillance & Early Warning System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    success = send_email(email, "Welcome to Vitanza - Verify Your Email", html_content)
    if success:
        return {"message": "Verification email sent successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send verification email")

@router.get("/verify")
async def verify_email(token: str, db=Depends(get_mongo_db)):
    """Verify user email with token"""
    verification = await db.email_verifications.find_one({
        "token": token,
        "expires_at": {"$gt": datetime.utcnow()}
    })
    
    if not verification:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    
    # Mark user as verified
    await db.users.update_one(
        {"email": verification["email"]},
        {"$set": {"email_verified": True, "verified_at": datetime.utcnow()}}
    )
    
    # Remove verification token
    await db.email_verifications.delete_one({"token": token})
    
    return {"message": "Email verified successfully"}

@router.post("/send-notification")
async def send_notification(notification: NotificationRequest, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "government", "ngo", "asha"))):
    """Send notification to users"""
    # Store notification in database
    notification_data = {
        "title": notification.title,
        "message": notification.message,
        "user_id": notification.user_id,
        "severity": notification.severity,
        "created_at": datetime.utcnow(),
        "sent_by": user.get("id")
    }
    
    await db.notifications.insert_one(notification_data)
    
    # Send email notification if user_id is provided
    if notification.user_id:
        user_doc = await db.users.find_one({"_id": notification.user_id})
        if user_doc and user_doc.get("email"):
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Vitanza Alert</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 30px; text-align: center; color: white; }}
                    .content {{ padding: 30px; }}
                    .alert {{ background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 20px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚨 Vitanza Alert</h1>
                    </div>
                    <div class="content">
                        <h2>{notification.title}</h2>
                        <div class="alert">
                            <p style="margin: 0; color: #dc2626; font-weight: bold;">{notification.message}</p>
                        </div>
                        <p style="color: #64748b;">Please take appropriate action as needed.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            send_email(user_doc["email"], f"Vitanza Alert: {notification.title}", html_content)
    
    return {"message": "Notification sent successfully"}

@router.get("/notifications")
async def get_notifications(user_id: Optional[str] = None, db=Depends(get_mongo_db), user=Depends(require_roles("admin", "officer", "government", "ngo", "asha"))):
    """Get notifications for user"""
    query = {}
    if user_id:
        query["user_id"] = user_id
    
    notifications = await db.notifications.find(query).sort("created_at", -1).limit(50).to_list(50)
    return notifications
