from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)

class Awareness(Base):
    __tablename__ = "awareness"
    id = Column(Integer, primary_key=True)
    key = Column(String)
    lang = Column(String)
    title = Column(String)
    body = Column(String)