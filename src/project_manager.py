import sqlite3
from sqlite3 import Connection
import os
import logging
from typing import Optional, List, Dict
from time import strftime
from datetime import datetime

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

TABLE_NAME = "Projects"

create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    user TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (name, user)
);
"""

def _connect_to_db() -> Optional[Connection]:
    db_path = os.environ.get(DATA_DIR, "./data")
    db_path = os.path.join(db_path, "projects.db")
    if not os.path.exists(db_path):
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with open(db_path, "w"):
                pass
        except Exception as e:
            logger.error(f"Failed to create database file: {e}")
            return None
    return sqlite3.connect(db_path)

def _ensure_projects_table(conn: Optional[Connection]=None):
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to create projects table: {e}")

def create_project(user: str, name: str, description: str = "") -> Optional[Dict]:
    """Create a new project for the user"""
    conn = _connect_to_db()
    if conn is None:
        return None
    
    _ensure_projects_table(conn)
    try:
        now = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {TABLE_NAME} (name, description, user, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, description, user, now, now)
        )
        project_id = cursor.lastrowid
        conn.commit()
        return {
            "id": project_id,
            "name": name,
            "description": description,
            "user": user,
            "created_at": now,
            "updated_at": now
        }
    except sqlite3.IntegrityError:
        logger.error(f"Project {name} already exists for user {user}")
        return None
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return None
    finally:
        conn.close()

def get_user_projects(user: str) -> List[Dict]:
    """Get all projects for a user"""
    conn = _connect_to_db()
    if conn is None:
        return []
    
    _ensure_projects_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, name, description, created_at, updated_at FROM {TABLE_NAME} WHERE user = ?",
            (user,)
        )
        projects = []
        for row in cursor.fetchall():
            projects.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            })
        return projects
    except Exception as e:
        logger.error(f"Failed to get user projects: {e}")
        return []
    finally:
        conn.close()

def get_project(user: str, project_id: int) -> Optional[Dict]:
    """Get a specific project by ID"""
    conn = _connect_to_db()
    if conn is None:
        return None
    
    _ensure_projects_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, name, description, created_at, updated_at FROM {TABLE_NAME} WHERE user = ? AND id = ?",
            (user, project_id)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
        return None
    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        return None
    finally:
        conn.close()

def update_project(user: str, project_id: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
    """Update a project's details"""
    conn = _connect_to_db()
    if conn is None:
        return False
    
    _ensure_projects_table(conn)
    try:
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if not updates:
            return True
        
        updates.append("updated_at = ?")
        params.extend([datetime.now().isoformat(), user, project_id])
        
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {TABLE_NAME} SET {', '.join(updates)} WHERE user = ? AND id = ?",
            params
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update project: {e}")
        return False
    finally:
        conn.close()

def delete_project(user: str, project_id: int) -> bool:
    """Delete a project"""
    conn = _connect_to_db()
    if conn is None:
        return False
    
    _ensure_projects_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {TABLE_NAME} WHERE user = ? AND id = ?",
            (user, project_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        return False
    finally:
        conn.close()
