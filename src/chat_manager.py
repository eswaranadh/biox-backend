import sqlite3
from sqlite3 import Connection
import os
import logging
from typing import Optional, List, Dict
from datetime import datetime

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

TABLE_NAME = "ProjectChats"

create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    user TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES Projects(id) ON DELETE CASCADE
);
"""

def _connect_to_db() -> Optional[Connection]:
    db_path = os.environ.get(DATA_DIR, "./data")
    db_path = os.path.join(db_path, "projects.db")  # Using same DB as projects
    if not os.path.exists(db_path):
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with open(db_path, "w"):
                pass
        except Exception as e:
            logger.error(f"Failed to create database file: {e}")
            return None
    return sqlite3.connect(db_path)

def _ensure_chats_table(conn: Optional[Connection]=None):
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to create chats table: {e}")

def create_chat(project_id: int, user: str, name: str = "New Chat") -> Optional[Dict]:
    """Create a new chat in a project"""
    conn = _connect_to_db()
    if conn is None:
        return None
    
    _ensure_chats_table(conn)
    try:
        now = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {TABLE_NAME} (project_id, name, user, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, name, user, now, now)
        )
        chat_id = cursor.lastrowid
        conn.commit()
        return {
            "id": chat_id,
            "project_id": project_id,
            "name": name,
            "user": user,
            "created_at": now,
            "updated_at": now
        }
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        return None
    finally:
        conn.close()

def get_project_chats(project_id: int, user: str) -> List[Dict]:
    """Get all chats in a project"""
    conn = _connect_to_db()
    if conn is None:
        return []
    
    _ensure_chats_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, name, created_at, updated_at FROM {TABLE_NAME} WHERE project_id = ? AND user = ? ORDER BY updated_at DESC",
            (project_id, user)
        )
        chats = []
        for row in cursor.fetchall():
            chats.append({
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "updated_at": row[3]
            })
        return chats
    except Exception as e:
        logger.error(f"Failed to get project chats: {e}")
        return []
    finally:
        conn.close()

def get_chat(chat_id: int, user: str) -> Optional[Dict]:
    """Get a specific chat"""
    conn = _connect_to_db()
    if conn is None:
        return None
    
    _ensure_chats_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, project_id, name, created_at, updated_at FROM {TABLE_NAME} WHERE id = ? AND user = ?",
            (chat_id, user)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "project_id": row[1],
                "name": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
        return None
    except Exception as e:
        logger.error(f"Failed to get chat: {e}")
        return None
    finally:
        conn.close()

def update_chat(chat_id: int, user: str, name: str) -> bool:
    """Update a chat's name"""
    conn = _connect_to_db()
    if conn is None:
        return False
    
    _ensure_chats_table(conn)
    try:
        now = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {TABLE_NAME} SET name = ?, updated_at = ? WHERE id = ? AND user = ?",
            (name, now, chat_id, user)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update chat: {e}")
        return False
    finally:
        conn.close()

def delete_chat(chat_id: int, user: str) -> bool:
    """Delete a chat"""
    conn = _connect_to_db()
    if conn is None:
        return False
    
    _ensure_chats_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {TABLE_NAME} WHERE id = ? AND user = ?",
            (chat_id, user)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete chat: {e}")
        return False
    finally:
        conn.close()
