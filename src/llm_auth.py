

from functools import wraps
import os
from typing import Optional, Tuple

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from src.constants import (
    AZURE_COMMUNITY,
    GPT_COMMUNITY,
    OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_MODEL,
    TOKEN_DAILY_LIMITATION
)
from src.datatypes import AuthTypeEnum, ModelConfig

def _parse_api_key(bearToken: str) -> str:
    if not bearToken:
        return ""
    bearToken = bearToken.strip()
    bearToken = bearToken.replace("Bearer ", "")
    return bearToken

def llm_get_auth_type(client_key: Optional[str]=None) -> AuthTypeEnum:
    if client_key is not None and len(client_key.strip()) > 0:
        return AuthTypeEnum.ClientOpenAI
    
    if os.environ.get("OPENAI_API_TYPE", "") == "azure":
        return AuthTypeEnum.ServerAzureOpenAI
    if len(os.environ.get("OPENAI_API_KEY", "")) > 0:
        return AuthTypeEnum.ServerOpenAI
    
    return AuthTypeEnum.Unknown

def llm_get_auth_token_limitation(auth_type: AuthTypeEnum) -> int:
    if auth_type is AuthTypeEnum.ClientOpenAI:
        return -1
    if auth_type is AuthTypeEnum.ServerAzureOpenAI or \
       auth_type is AuthTypeEnum.ServerOpenAI:
        return os.environ.get(TOKEN_DAILY_LIMITATION, -1)
    
    return -1

def llm_get_user_name_and_model(
        client_key: Optional[str]=None,
        session_id: Optional[str]=None,
        model: Optional[str]=None
    ) -> Tuple[str, str]:
    auth_type = llm_get_auth_type(client_key=client_key)
    if auth_type == AuthTypeEnum.ClientOpenAI:
        return session_id, model
    mod = llm_get_model_by_AuthType(auth_type, model)
    return (
        AZURE_COMMUNITY if auth_type == AuthTypeEnum.ServerAzureOpenAI else GPT_COMMUNITY,
        mod
    )

def llm_get_model_by_AuthType(auth_type: AuthTypeEnum | None, model: Optional[str]):
    default_model = model if model is not None and len(model) > 0 else "gpt-3.5-turbo"
    if auth_type is None or auth_type == AuthTypeEnum.ClientOpenAI:
        return default_model
    
    model = os.environ.get(OPENAI_MODEL, default_model)
    return model if len(model) > 0 else default_model

def llm_get_user_name_by_AuthType(auth_type: AuthTypeEnum, session_id: str) -> str:
    if auth_type == AuthTypeEnum.ClientOpenAI:
        return session_id
    return AZURE_COMMUNITY if auth_type == AuthTypeEnum.ServerAzureOpenAI \
        else GPT_COMMUNITY

def llm_get_auth_key_by_AuthType(auth_type: AuthTypeEnum, modelConfig: ModelConfig):
    at = auth_type.value
    if at[:6] == "Server":
        return os.environ.get(OPENAI_API_KEY, "")
    else:
        return modelConfig.openai_api_key

def llm_get_client_auth(client_key: str | None) -> str | None:
    # try to parse bearer key first
    key = _parse_api_key(client_key)
    
    return key if key is not None else ""

def llm_get_auth_key(client_key: Optional[str]=None):
    auth = None
    if client_key is None or len(client_key) > 0:
        auth = _parse_api_key(client_key)
    
    if auth is not None and len(auth) > 0:
        return auth
    
    # OPENAI_API_KEY is provided by server
    if OPENAI_API_KEY in os.environ and os.environ[OPENAI_API_KEY]:
        auth = os.environ[OPENAI_API_KEY]
    
    return auth

def save_and_restore_openai_type(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "auth_type" in kwargs:
            auth_type = kwargs.get("auth_type", AuthTypeEnum.Unknown)
        else:
            auth_type = args[0]
        openai_type = ""
        if auth_type == AuthTypeEnum.ClientOpenAI:
            openai_type = os.environ.get(OPENAI_API_TYPE, "")
            os.environ[OPENAI_API_TYPE] = ""
        result = func(*args, **kwargs)
        if auth_type == AuthTypeEnum.ClientOpenAI and len(openai_type) > 0:
            os.environ[OPENAI_API_TYPE] = openai_type
        return result
    return wrapper
    
@save_and_restore_openai_type
def _get_embedding_function(
    auth_type: AuthTypeEnum,
    client_key: str,
    model: str,
):
    if auth_type == AuthTypeEnum.ClientOpenAI:
        return OpenAIEmbeddings(
            api_key=client_key,
            model=model,
        )
    elif auth_type == AuthTypeEnum.ServerOpenAI:
        return OpenAIEmbeddings(
            api_key=llm_get_auth_key(),
        )
    elif auth_type == AuthTypeEnum.ServerAzureOpenAI:
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return AzureOpenAIEmbeddings(
            api_key=llm_get_auth_key(),
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            model=model,
        )
    else:
        return None

def llm_get_embedding_function(
        client_key: Optional[str]=None,
        model: Optional[str] = "text-embedding-ada-002",
    ) -> OpenAIEmbeddings | AzureOpenAIEmbeddings | None:
    auth_type = llm_get_auth_type(client_key)
    return _get_embedding_function(
        auth_type=auth_type,
        client_key=client_key,
        model=model,
    )
    