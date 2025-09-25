import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase Admin SDK
def initialize_firebase():
    if not firebase_admin._apps:
        # Use service account key file or environment variables
        if os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'):
            # If service account key is provided as JSON string
            import json
            service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
            cred = credentials.Certificate(service_account_info)
        else:
            # Use default credentials (for production with proper IAM roles)
            cred = credentials.ApplicationDefault()
        
        firebase_admin.initialize_app(cred)
    
    return firebase_admin.get_app()

# Initialize Firebase
app = initialize_firebase()

# Get Firebase services
def get_firebase_auth():
    return auth

def get_firestore_client():
    return firestore.client()

# Verify Firebase token
async def verify_firebase_token(token: str):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise Exception(f"Invalid token: {str(e)}")

# Create custom token
async def create_custom_token(uid: str, additional_claims: dict = None):
    try:
        return auth.create_custom_token(uid, additional_claims)
    except Exception as e:
        raise Exception(f"Failed to create custom token: {str(e)}")
