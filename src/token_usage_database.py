
import sqlite3
from sqlite3 import Connection
import os
from time import strftime
from typing import Optional
import logging

from src.constants import (
    AZURE_COMMUNITY,
    DATA_DIR,
    GPT_COMMUNITY
)

logger = logging.getLogger(__name__)

TABLE_NAME = "LLMTokenUsage"

create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    date TEXT NOT NULL,
    hour TEXT NOT NULL,
    user TEXT NOT NULL,
    model TEXT not NULL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    UNIQUE (user, model, date, hour)
);
"""

def _ensure_azure_token_usage_tables(conn: Optional[Connection]=None):
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
    except Exception as e:
        logger.error(e)
    pass

def _connect_to_db():
    db_path = os.environ.get(DATA_DIR, "./data")
    db_path = os.path.join(db_path, "token_usage.db")
    if not os.path.exists(db_path):
        try:
            with open(db_path, "w"):
                pass
        except Exception as e:
            logger.error(e)
            return None
    return sqlite3.connect(db_path)

upsert_usage_query = f"""
insert into {TABLE_NAME}(user, session_id, model, date, hour, completion_tokens, prompt_tokens, total_tokens) 
values (?,?,?,?,?,?,?,?) 
on conflict(user, model, date, hour)
do update set completion_tokens=completion_tokens+excluded.completion_tokens,
prompt_tokens=prompt_tokens+excluded.prompt_tokens,
total_tokens=total_tokens+excluded.total_tokens;
"""
def _update_token_usage(conn: Connection, user: str, session_id: str, model: str, token_usage: dict):
    date=strftime("%Y-%m-%d")
    hour=strftime("%H")
    cursor = conn.cursor()        
    cursor.execute(upsert_usage_query,
                    (user, session_id, model, date, hour,
                    token_usage["completion_tokens"],
                    token_usage["prompt_tokens"],
                    token_usage["total_tokens"]))
    conn.commit()

def update_token_usage(user: str, session_id: str, model: str, token_usage: dict):
    conn = _connect_to_db()
    if conn is None:
        return
    _ensure_azure_token_usage_tables(conn)
    try:
        _update_token_usage(conn, user, session_id, model, token_usage)
    except Exception as e:
        logger.error(e)
    finally:
        conn.close()

select_usage_with_model_query = f"""
select completion_tokens, prompt_tokens, total_tokens from {TABLE_NAME}
where user=? and model=? and date=?;
"""
select_usage_query = f"""
select completion_tokens, prompt_tokens, total_tokens from {TABLE_NAME}
where user=? and date=?;
"""
def get_token_usage(user: str, model: Optional[str]=None):
    date = strftime("%Y-%m-%d")
    conn = _connect_to_db()
    if conn is None:
        return None
    _ensure_azure_token_usage_tables(conn)
    try:
        cursor = conn.cursor()
        if model is not None:
            cursor.execute(select_usage_with_model_query, (user, model, date))
        else:
            cursor.execute(select_usage_query, (user, date))
        result = cursor.fetchall()
        cursor.close()

        res = [0, 0, 0]
        result = result if result is not None else []
        for item in result:
            res[0] += item[0]
            res[1] += item[1]
            res[2] += item[2]

        return {
            "completion_tokens": res[0], 
            "prompt_tokens": res[1],
            "total_tokens": res[2],
        } 
    except Exception as e:
        logger.error(e)
        return None
    finally:
        conn.close()
    



