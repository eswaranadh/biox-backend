import os
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any

import threading
from threading import RLock

from src.conversation_session import (
    ConversationSession,
    defaultModelConfig,
)

logger = logging.getLogger(__name__)

rlock = RLock()

MAX_AGE = 3 * 24 * 3600 * 1000  # 3 days

conversationsDict = {}


def get_conversation_id(session_id: str, project_id: Optional[int] = None) -> str:
    """Generate a unique conversation ID combining session and project"""
    if project_id is not None:
        return f"{session_id}_{project_id}"
    return session_id


def initialize_conversation(sessionId: str, modelConfig: dict, project_id: Optional[int] = None):
    rlock.acquire()
    try:
        conversation_id = get_conversation_id(sessionId, project_id)
        conversationsDict[conversation_id] = ConversationSession(
            sessionId=sessionId,
            modelConfig=modelConfig,
        )
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def has_conversation(sessionId: str, project_id: Optional[int] = None) -> bool:
    rlock.acquire()
    try:
        conversation_id = get_conversation_id(sessionId, project_id)
        return conversation_id in conversationsDict
    finally:
        rlock.release()


def get_conversation(
        sessionId: str, 
        modelConfig: Optional[Dict]=None,
        project_id: Optional[int] = None
    ) -> Optional[ConversationSession]:
    rlock.acquire()
    try:
        conversation_id = get_conversation_id(sessionId, project_id)
        if conversation_id not in conversationsDict:
            initialize_conversation(
                sessionId,
                modelConfig=defaultModelConfig.copy() \
                    if modelConfig is None else modelConfig,
                project_id=project_id
            )
        return conversationsDict[conversation_id]
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def remove_conversation(sessionId: str, project_id: Optional[int] = None):
    rlock.acquire()
    try:
        conversation_id = get_conversation_id(sessionId, project_id)
        if conversation_id not in conversationsDict:
            return
        del conversationsDict[conversation_id]
    except Exception as e:
        logger.error(e)
    finally:
        rlock.release()


def chat(
    sessionId: str,
    messages: List[str],
    useRAG: bool,
    useKG: bool,
    useAutoAgent: Optional[bool] = None,
    ragConfig: Optional[Dict]=None,
    kgConfig: Optional[Dict]=None,
    oncokbConfig: Optional[dict] = None,
    modelConfig: Optional[Dict] = None,
    project_id: Optional[int] = None,
):
    rlock.acquire()
    useAutoAgent = False if useAutoAgent is None else useAutoAgent
    try:
        conversation = get_conversation(sessionId=sessionId, project_id=project_id)
        logger.info(
            f"get conversation for session id {sessionId}, "
            "type of conversation is ConversationSession "
            f"{isinstance(conversation, ConversationSession)}"
        )
        return conversation.chat(
            messages=messages,
            ragConfig=ragConfig,
            useRAG=useRAG,
            kgConfig=kgConfig,
            useKG=useKG,
            useAutoAgent=useAutoAgent,
            oncokbConfig=oncokbConfig,
            modelConfig=modelConfig,
        )
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def recycle_conversations():
    logger.info(f"[recycle] - {threading.get_native_id()} recycle_conversation")
    rlock.acquire()
    now = datetime.now().timestamp() * 1000  # in milliseconds
    sessionsToRemove: List[str] = []
    try:
        for conversation_id in conversationsDict.keys():
            sessionId, _, project_id = conversation_id.partition("_")
            conversation = get_conversation(sessionId=sessionId, project_id=int(project_id) if project_id else None)
            assert conversation is not None
            logger.info(
                f"[recycle] sessionId is {sessionId}, "
                f"refreshAt: {conversation.sessionData.refreshedAt}, "
                f"maxAge: {conversation.sessionData.maxAge}"
            )
            if conversation.sessionData.refreshedAt + conversation.sessionData.maxAge < now:
                sessionsToRemove.append(conversation_id)
        for conversation_id in sessionsToRemove:
            remove_conversation(sessionId=conversation_id, project_id=None)
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()
