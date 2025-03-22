from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String, create_engine, select, delete
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from datetime import datetime
import json
import os

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import AsyncAttrs

import aiofiles.os
import asyncio

class Base(AsyncAttrs, DeclarativeBase):
    pass

# Detection model defenition
class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    video_path: Mapped[Optional[str]] = mapped_column(nullable=True)
    identities: Mapped[Optional[str]] = mapped_column(nullable=True)
    strict_eval_safe: Mapped[Optional[str]] = mapped_column(nullable=True)
    relaxed_eval_safe: Mapped[Optional[str]] = mapped_column(nullable=True)
    face_results: Mapped[Optional[str]] = mapped_column(nullable=True)
    simultaneous_face_ids: Mapped[Optional[str]] = mapped_column(nullable=True)
    simultaneous_body_ids: Mapped[Optional[str]] = mapped_column(nullable=True)
    face_results: Mapped[Optional[str]] = mapped_column(nullable=True)
    face_amt_eval: Mapped[Optional[str]] = mapped_column(nullable=True)

class UserPreferences(Base):
    __tablename__ = "evaluation_mode"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # True = strict, False = relaxed
    evaluation_mode: Mapped[bool] = mapped_column(default=False)
    # True = Dark mode, False = light mode
    color_theme: Mapped[bool] = mapped_column(default=False)

# Create an engine and initialize the database
#engine = create_engine("sqlite:///detections.db", echo=False)
engine = create_async_engine("sqlite+aiosqlite:///detections.db", echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def initialize():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(initialize())

async def add_detection(async_session: async_sessionmaker[AsyncSession], 
                        identities,
                        strict_eval_safe,
                        relaxed_eval_safe,
                        video_path,
                        face_results,
                        simultaneous_face_ids,
                        face_amt_eval,
                        simultaneous_body_ids,
                        ) -> None: 
    async with async_session() as session:
        async with session.begin():
            session.add(Detection(
                video_path=video_path,
                identities= identities,
                strict_eval_safe= strict_eval_safe,
                relaxed_eval_safe= relaxed_eval_safe,
                face_results= face_results,
                simultaneous_face_ids= simultaneous_face_ids,
                face_amt_eval= face_amt_eval,
                simultaneous_body_ids= simultaneous_body_ids
            ))

        await session.commit()

# get user preferences
async def get_user_preferences():
    async with async_session() as session:
        result = await session.execute(select(UserPreferences))
        preference = result.scalars().first()
        # If appplication launches the very first time
        if preference is None:
            preference = UserPreferences(evaluation_mode=False, color_theme=False)
            session.add(preference)
            await session.commit()
        return preference

async def set_user_preferences(evaluation_mode: bool = None, color_theme: bool = None) -> bool:
    async with async_session() as session:
        # Try to fetch existing preferences
        result = await session.execute(select(UserPreferences))
        preferences = result.scalars().first()
        # Update prefence if passed
        preferences.evaluation_mode = preferences.evaluation_mode if evaluation_mode == None else evaluation_mode
        preferences.color_theme = preferences.color_theme if color_theme == None else color_theme
        
        await session.commit()
        return True

# update video path
async def update_video_path(async_session: async_sessionmaker[AsyncSession], detection_id: str, video_path: str, video_dir: str = "saved_clips") -> None:
    
    video_path = os.path.relpath(video_path, video_dir)
    
    async with async_session() as session:
        result = await session.execute(select(Detection).where(Detection.id == detection_id))
        detection = result.scalars().first()

        if detection:
            detection.video_path = video_path
            await session.commit()

# Get all detections
async def get_detections(async_session: async_sessionmaker[AsyncSession]) -> List[Detection]:
    async with async_session() as session:
        result = await session.execute(select(Detection).order_by(Detection.timestamp.desc()))
        return result.scalars().all()

# Get particular detections using primary key
async def get_detection(detection_id: str, async_session: async_sessionmaker[AsyncSession]) -> Optional[Detection]:
    async with async_session() as session:
        result = await session.execute(select(Detection).where(Detection.id == detection_id))
        return result.scalars().first()

# Delete particular detection via primary key
async def delete_detection(async_session: async_sessionmaker[AsyncSession], detection_id: int) -> bool:
    async with async_session() as session:
        detection = await session.get(Detection, detection_id)
        if detection:
            try:
                os.remove(detection.video_path)
            except Exception as e:
                print(f"Error deleting video file: {e}")

            await session.delete(detection)
            await session.commit()
            return True
        return False

# Delte all detections and video files
async def delete_all_detections(async_session: async_sessionmaker[AsyncSession]) -> None:
    async with async_session() as session:
        result = await session.execute(select(Detection))
        detections = result.scalars().all()

        for detection in detections:
            try:
                await asyncio.to_thread(os.remove, detection.video_path)
            except Exception as e:
                print(f"Error deleting video file {detection.video_path}: {e}")

        await session.execute(delete(Detection))
        await session.commit()
        return
