
import base64
import os
from typing import List, Optional, Tuple

from src.constants import TOKEN_DAILY_LIMITATION
from src.datatypes import AuthTypeEnum
from src.llm_auth import llm_get_auth_type, llm_get_user_name_and_model
from src.token_usage_database import get_token_usage

def get_rag_agent_prompts() -> List[str]:
    return [
        "The user has provided additional background information from scientific "
        "articles.",
        "Take the following statements into account and specifically comment on "
        "consistencies and inconsistencies with all other information available to "
        "you: {statements}",
    ]
    
def need_restrict_usage(client_key: str, model: str) -> Tuple[bool, int]:
    auth_type = llm_get_auth_type(client_key=client_key)
    if auth_type == AuthTypeEnum.ClientOpenAI or \
       auth_type == AuthTypeEnum.ClientWASM:
        return False, -1
    limitation = int(os.environ.get(TOKEN_DAILY_LIMITATION, -1))
    if limitation < 0:
        return False, -1
    user_name, actual_model = llm_get_user_name_and_model(
        client_key=client_key,
        session_id=None,
        model=model,
    )
    token_usage = get_token_usage(user_name, actual_model)
    return token_usage["total_tokens"] >= limitation, limitation
