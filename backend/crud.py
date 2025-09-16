from typing import Optional

# Simple in-memory user store for demo purposes
_users_db: dict[str, dict] = {}

def get_user_by_username(db, username: str) -> Optional[object]:
    return _users_db.get(username)

def create_user(db, user, hashed_password: str):
    user_obj = {
        "id": len(_users_db) + 1,
        "username": user.username,
        "hashed_password": hashed_password,
        "role": user.role,
    }
    _users_db[user.username] = user_obj
    return user_obj

