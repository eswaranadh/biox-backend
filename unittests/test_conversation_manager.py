import os
from unittest.mock import patch
from src.constants import AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_VERSION, OPENAI_DEPLOYMENT_NAME, OPENAI_MODEL
from src.conversation_manager import (
    get_conversation, 
    has_conversation,
    initialize_conversation, 
    remove_conversation,
)
from src.conversation_session import defaultModelConfig
from src.llm_auth import _parse_api_key

def test_parse_api_key():
    res = _parse_api_key("Bearer balahbalah")
    assert res == "balahbalah"

#@patch("src.conversation_session.AzureGptConversation")
@patch.dict(os.environ, {
    OPENAI_API_KEY: "abcdefg",
    OPENAI_API_VERSION: "2024-02-01",
    OPENAI_MODEL: "gpt-4",
})
@patch("src.conversation_session.GptConversation")
def test_get_conversation(mock_GptConversation):
    modelConfig = {
        **defaultModelConfig,
        "chatter_type": "ServerOpenAI",
    }
    conversation = get_conversation(
        sessionId="balahbalah", modelConfig=modelConfig,
    )
    assert conversation is not None
    assert conversation.sessionData.sessionId == "balahbalah"
    assert conversation.chatter is not None
    assert has_conversation("balahbalah") 

@patch.dict(os.environ, {
    OPENAI_API_KEY: "abcdefg",
    OPENAI_API_VERSION: "2024-02-01",
    OPENAI_MODEL: "gpt-4",
})
@patch("src.conversation_session.GptConversation")
def test_remove_conversation(mock_GptConversation):
    sessionId = "test"
    assert not has_conversation(sessionId)
    initialize_conversation(
        sessionId=sessionId,
        modelConfig={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "sendMemory": True,
            "historyMessageCount": 4,
            "compressMessageLengthThreshold": 2000,
            "chatter_type": "ServerOpenAI",
        }
    )
    assert has_conversation(sessionId)
    remove_conversation(sessionId)
    assert not has_conversation(sessionId)


