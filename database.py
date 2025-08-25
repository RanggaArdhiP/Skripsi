from sqlalchemy import Column, Integer, String, BLOB, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import settings

Base = declarative_base()
engine = settings.engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String, index=True)
    source_path = Column(String)
    detected_image = Column(BLOB)

Base.metadata.create_all(bind=engine)
