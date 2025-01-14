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


def initialize_conversation(sessionId: str, modelConfig: dict):
    rlock.acquire()
    try:
        conversationsDict[sessionId] = ConversationSession(
            sessionId=sessionId,
            modelConfig=modelConfig,
        )
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def has_conversation(sessionId: str) -> bool:
    rlock.acquire()
    try:
        return sessionId in conversationsDict
    finally:
        rlock.release()


def get_conversation(
        sessionId: str, 
        modelConfig: Optional[Dict]=None
    ) -> Optional[ConversationSession]:
    rlock.acquire()
    try:
        if sessionId not in conversationsDict:
            initialize_conversation(
                sessionId,
                modelConfig=defaultModelConfig.copy() \
                    if modelConfig is None else modelConfig
            )
        return conversationsDict[sessionId]
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def remove_conversation(sessionId: str):
    rlock.acquire()
    try:
        if sessionId not in conversationsDict:
            return
        del conversationsDict[sessionId]
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
):
    rlock.acquire()
    useAutoAgent = False if useAutoAgent is None else useAutoAgent
    try:
        conversation = get_conversation(sessionId=sessionId)
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
        for sessionId in conversationsDict.keys():
            conversation = get_conversation(sessionId=sessionId)
            assert conversation is not None
            logger.info(
                f"[recycle] sessionId is {sessionId}, "
                f"refreshAt: {conversation.sessionData.refreshedAt}, "
                f"maxAge: {conversation.sessionData.maxAge}"
            )
            if conversation.sessionData.refreshedAt + conversation.sessionData.maxAge < now:
                sessionsToRemove.append(sessionId)
        for sessionId in sessionsToRemove:
            remove_conversation(sessionId)
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()
