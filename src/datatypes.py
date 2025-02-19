from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel

class ConnectionArgs(BaseModel):
    host: str
    port: str
    user: Optional[str]=None
    password: Optional[str]=None

class RagConfig(BaseModel):
    splitByChar: bool
    chunkSize: int
    overlapSize: int
    resultNum: int
    connectionArgs: ConnectionArgs
    docIdsWorkspace: Optional[List[str]]=None
    description: Optional[str]=None

class KGConfig(BaseModel):
    resultNum: int
    connectionArgs: ConnectionArgs
    description: Optional[str]=None

class RagNewDocumentPostModel(BaseModel):
    tmpFile: str
    filename: str
    ragConfig: str

class RagAllDocumentsPostModel(BaseModel):
    connectionArgs: ConnectionArgs
    docIds: Optional[List[str]] = None

class RagDocumentDeleteModel(BaseModel):
    connectionArgs: ConnectionArgs
    docId: str
    docIds: Optional[List[str]] = None

class RagConnectionStatusPostModel(BaseModel):
    connectionArgs: ConnectionArgs

class KgConnectionStatusPostModel(BaseModel):
    connectionArgs: ConnectionArgs

class Message(BaseModel):
    role: str
    content: str

class OncoKBConfig(BaseModel):
    useOncoKB: Optional[bool]=False
    description: Optional[str] = None

class TokenUsagePostModel(BaseModel):
    model: Optional[str] = None
    session_id: Optional[str] = None

class ChatCompletionsPostModel(BaseModel):
    session_id: str
    chat_id: Optional[int] = None
    messages: List[Message]
    model: str
    temperature: int
    presence_penalty: int
    frequency_penalty: int
    top_p: int
    useRAG: bool
    ragConfig: Optional[RagConfig]=None
    useKG: bool
    kgConfig: Optional[KGConfig]=None
    stream: Optional[bool]=None
    oncokbConfig: Optional[OncoKBConfig]=None
    useAutoAgent: Optional[bool]=None
    project_id: Optional[int]=None

class AuthTypeEnum(Enum):
    Unknown = "Unknown"
    ServerAzureOpenAI = "ServerAzureOpenAI"
    ServerOpenAI = "ServerOpenAI"
    ClientOpenAI = "ClientOpenAI"
    ClientWASM = "ClientWASM"

class ModelConfig(BaseModel):
    model: str
    temperature: float
    presence_penalty: int
    frequency_penalty: int
    top_p: Optional[int]=None
    chatter_type: Optional[AuthTypeEnum]=AuthTypeEnum.Unknown
    openai_api_key: Optional[str]=None

# Project-related models
class ProjectCreateModel(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectUpdateModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectIdModel(BaseModel):
    project_id: int

# Chat-related models
class ChatCreateModel(BaseModel):
    project_id: int
    name: str = "New Chat"

class ChatUpdateModel(BaseModel):
    name: str

class ChatModel(BaseModel):
    id: int
    project_id: int
    name: str
    created_at: str
    updated_at: str
