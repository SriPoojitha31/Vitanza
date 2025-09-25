from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from bson import ObjectId
from mongo import get_mongo_db
import os

router = APIRouter()

JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")
JWT_ALG = "HS256"
JWT_EXP_MIN = int(os.getenv("JWT_EXP_MIN", "240"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALLOWED_ROLES = {"community", "worker", "officer", "admin", "ngo", "government", "asha"}

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    displayName: Optional[str] = None
    role: str = "community"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: str
    email: EmailStr
    displayName: Optional[str] = None
    role: str

def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    return pwd_ctx.verify(pw, hashed)

def create_token(sub: str, role: str) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": sub,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXP_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

@router.post("/register", response_model=UserOut)
async def register(body: UserRegister, db=Depends(get_mongo_db)):
    # Normalize and validate role
    role = (body.role or "community").lower()
    role_map = {"gov": "government", "ngo": "ngo", "asha": "asha", "officer": "officer", "worker": "worker", "admin": "admin", "community": "community"}
    role = role_map.get(role, role)
    if role not in ALLOWED_ROLES:
        role = "community"
    existing = await db.users.find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = {
        "email": body.email,
        "password": hash_password(body.password),
        "displayName": body.displayName or body.email,
        "role": role,
        "email_verified": False,
        "createdAt": datetime.utcnow(),
    }
    res = await db.users.insert_one(doc)
    
    # Send verification email
    try:
        from routes.email import send_email
        import secrets
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        await db.email_verifications.insert_one({
            "email": body.email,
            "token": token,
            "expires_at": expires_at,
            "created_at": datetime.utcnow()
        })
        
        # Send verification email
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Welcome to Vitanza</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center; color: white; }}
                .content {{ padding: 40px 30px; }}
                .button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 20px 0; }}
                .footer {{ background: #f8fafc; padding: 30px; text-align: center; color: #64748b; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">🌊 Vitanza</div>
                    <p style="margin: 0; opacity: 0.9;">Smart Health Surveillance System</p>
                </div>
                <div class="content">
                    <h2 style="color: #1e293b; margin-top: 0;">Welcome to Vitanza!</h2>
                    <p style="color: #475569; line-height: 1.6;">Thank you for joining our community health monitoring platform. Please verify your email address to complete your registration.</p>
                    <div style="text-align: center;">
                        <a href="https://vitanza.buzz/verify?token={token}" class="button">Verify Email Address</a>
                    </div>
                    <p style="color: #64748b; font-size: 14px; margin-top: 30px;">This verification link will expire in 24 hours.</p>
                </div>
                <div class="footer">
                    <p>© 2024 Vitanza. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        send_email(body.email, "Welcome to Vitanza - Verify Your Email", html_content)
    except Exception as e:
        print(f"Failed to send verification email: {e}")
    
    return UserOut(id=str(res.inserted_id), email=doc["email"], displayName=doc["displayName"], role=doc["role"]) 

@router.post("/login", response_model=TokenOut)
async def login(body: UserLogin, db=Depends(get_mongo_db)):
    user = await db.users.find_one({"email": body.email})
    if not user or not verify_password(body.password, user.get("password", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(str(user["_id"]), user.get("role", "community"))
    return TokenOut(access_token=token)

async def get_current_user(authorization: Optional[str] = Header(None), db=Depends(get_mongo_db)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user = await db.users.find_one({"_id": ObjectId(uid)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {"id": str(user["_id"]), "email": user["email"], "role": user.get("role", "community"), "displayName": user.get("displayName")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_roles(*allowed: str):
    async def dep(user=Depends(get_current_user)):
        if user.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dep

@router.get("/me")
async def me(user=Depends(get_current_user)):
    return user


