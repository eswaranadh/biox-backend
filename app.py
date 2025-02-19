import json
from typing import Optional, Any, List, Annotated, Dict

import uvicorn
from fastapi import FastAPI, Header, Request
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import atexit
import logging
import os
from pymilvus import MilvusException
import pymilvus
from src.constants import (
    ARGS_CONNECTION_ARGS,
    ERROR_EXCEEDS_TOKEN_LIMIT,
    ERROR_MILVUS_CONNECT_FAILED,
    ERROR_MILVUS_UNKNOWN,
    ERROR_OK,
    ERROR_UNKNOWN,
    ERRSTR_MILVUS_CONNECT_FAILED,
)
from src.conversation_manager import (
    chat,
    has_conversation,
    initialize_conversation,
)

from src.datatypes import (
    ChatCompletionsPostModel,
    AuthTypeEnum, 
    KgConnectionStatusPostModel, 
    RagAllDocumentsPostModel, 
    RagConnectionStatusPostModel, 
    RagDocumentDeleteModel, 
    RagNewDocumentPostModel,
    TokenUsagePostModel,
    ProjectCreateModel,
    ProjectUpdateModel,
    ProjectIdModel,
    ChatCreateModel,
    ChatUpdateModel,
    ChatModel,
)
from src.document_embedder import (
    get_all_documents,
    get_connection_status as get_vectorstore_connection_status,
    new_embedder_document,
    remove_document,
)
from src.kg_agent import get_connection_status as get_kg_connection_status
from src.llm_auth import (
    llm_get_auth_token_limitation,
    llm_get_auth_type,
    llm_get_client_auth,
    llm_get_embedding_function,
    llm_get_user_name_and_model,
)
from src.job_recycle_conversations import run_scheduled_job_continuously
from src.project_manager import (
    create_project,
    get_user_projects,
    get_project,
    update_project,
    delete_project,
)
from src.chat_manager import (
    create_chat,
    get_project_chats,
    get_chat,
    update_chat,
    delete_chat,
)
from src.token_usage_database import get_token_usage
from src.utils import need_restrict_usage

# prepare logger
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler("./logs/app.log")
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)


# run scheduled job: recycle unused session
cease_event = run_scheduled_job_continuously()


def onExit():
    cease_event.set()


atexit.register(onExit)

load_dotenv()
app = FastAPI(
    # Initialize FastAPI cache with in-memory backend
    title="Biochatter server API",
    version="0.3.1",
    description="API to interact with biochatter server",
    debug=True,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


DEFAULT_RAGCONFIG = {
    "splitByChar": True,
    "chunkSize": 1000,
    "overlapSize": 0,
    "resultNum": 3,
}
RAG_KG = "KG"
RAG_VECTORSTORE = "VS"
KG_VECTORSTORE = "KGVS"


def process_connection_args(rag: str, connection_args: dict) -> dict:
    if rag == RAG_VECTORSTORE:
        if connection_args.get("host", "").lower() == "local":
            connection_args["host"] = (
                "127.0.0.1" if "HOST" not in os.environ else os.environ["HOST"]
            )
        port = connection_args.get("port", "19530")
        connection_args["port"] = f"{port}"
    elif rag == RAG_KG:
        if connection_args.get("host", "").lower() == "local":
            connection_args["host"] = (
                "127.0.0.1" if "KGHOST" not in os.environ else os.environ["KGHOST"]
            )
        port = connection_args.get("port", "7687")
        connection_args["port"] = f"{port}"
    elif rag == KG_VECTORSTORE:
        if connection_args.get("host", "").lower() == "local":
            connection_args["host"] = (
                "127.0.0.1" if "KGHOST" not in os.environ else os.environ["KGHOST"]
            )
        port = connection_args.get("port", "19530")
        connection_args["port"] = f"{port}"
    return connection_args

def extract_and_process_params_from_json_body(
    json: Optional[Dict], name: str, defaultVal: Optional[Any]=None,
) -> Optional[Any]:
    if not json:
        return defaultVal
    val = json.get(name, defaultVal)
    return val

@app.post("/v1/chats/create", description="Create a new chat in a project")
def createChat(
    authorization: Annotated[str | None, Header()],
    item: ChatCreateModel,
):
    try:
        auth = llm_get_client_auth(authorization)
        user, _ = llm_get_user_name_and_model(auth)
        
        # Verify project exists and belongs to user
        project = get_project(user, item.project_id)
        if project is None:
            return {"error": "Project not found", "code": ERROR_UNKNOWN}
        
        chat = create_chat(item.project_id, user, item.name)
        if chat is None:
            return {"error": "Failed to create chat", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK, "chat": chat}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.get("/v1/projects/{project_id}/chats", description="Get all chats in a project")
def getProjectChats(
    authorization: Annotated[str | None, Header()],
    project_id: int,
):
    try:
        auth = llm_get_client_auth(authorization)
        user, _ = llm_get_user_name_and_model(auth)
        
        # Verify project exists and belongs to user
        project = get_project(user, project_id)
        if project is None:
            return {"error": "Project not found", "code": ERROR_UNKNOWN}
        
        chats = get_project_chats(project_id, user)
        return {"code": ERROR_OK, "chats": chats}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.get("/v1/chats/{chat_id}", description="Get a specific chat")
def getChat(
    authorization: Annotated[str | None, Header()],
    chat_id: int,
):
    try:
        auth = llm_get_client_auth(authorization)
        user, _ = llm_get_user_name_and_model(auth)
        chat = get_chat(chat_id, user)
        if chat is None:
            return {"error": "Chat not found", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK, "chat": chat}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.put("/v1/chats/{chat_id}", description="Update a chat")
def updateChat(
    authorization: Annotated[str | None, Header()],
    chat_id: int,
    item: ChatUpdateModel,
):
    try:
        auth = llm_get_client_auth(authorization)
        user, _ = llm_get_user_name_and_model(auth)
        success = update_chat(chat_id, user, item.name)
        if not success:
            return {"error": "Chat not found or update failed", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.delete("/v1/chats/{chat_id}", description="Delete a chat")
def deleteChat(
    authorization: Annotated[str | None, Header()],
    chat_id: int,
):
    try:
        auth = llm_get_client_auth(authorization)
        user, _ = llm_get_user_name_and_model(auth)
        success = delete_chat(chat_id, user)
        if not success:
            return {"error": "Chat not found or deletion failed", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.post("/v1/chat/completions", description="chat completions")
async def chat_completions(request: Request):
    try:
        authorization = request.headers.get("Authorization")
        auth = llm_get_client_auth(authorization)
        auth_type = llm_get_auth_type(auth)
        jsonBody = await request.json()

        sessionId = extract_and_process_params_from_json_body(
            jsonBody, "session_id", defaultVal=""
        )
        chat_id = extract_and_process_params_from_json_body(
            jsonBody, "chat_id", defaultVal=None
        )
        messages = extract_and_process_params_from_json_body(
            jsonBody, "messages", defaultVal=[]
        )
        model = extract_and_process_params_from_json_body(
            jsonBody, "model", defaultVal="gpt-3.5-turbo"
        )
        temperature = extract_and_process_params_from_json_body(
            jsonBody, "temperature", defaultVal=0.7
        )
        presence_penalty = extract_and_process_params_from_json_body(
            jsonBody, "presence_penalty", defaultVal=0
        )
        frequency_penalty = extract_and_process_params_from_json_body(
            jsonBody, "frequency_penalty", defaultVal=0
        )
        top_p = extract_and_process_params_from_json_body(
            jsonBody, "top_p", defaultVal=1
        )
        useRAG = extract_and_process_params_from_json_body(
            jsonBody, "useRAG", defaultVal=False
        )
        useKG = extract_and_process_params_from_json_body(
            jsonBody, "useKG", defaultVal=False
        )
        ragConfig = extract_and_process_params_from_json_body(
            jsonBody, "ragConfig", defaultVal=DEFAULT_RAGCONFIG.copy()
        )
        kgConfig = extract_and_process_params_from_json_body(
            jsonBody, "kgConfig", defaultVal={}
        )
        useAutoAgent = extract_and_process_params_from_json_body(
            jsonBody, "useAutoAgent", defaultVal=False
        )
        oncokbConfig = extract_and_process_params_from_json_body(
            jsonBody, "oncokbConfig", defaultVal={}
        )
        project_id = extract_and_process_params_from_json_body(
            jsonBody, "project_id", defaultVal=None
        )

        # If chat_id is provided, verify it exists and belongs to the user
        if chat_id is not None:
            user, _ = llm_get_user_name_and_model(auth)
            chat = get_chat(chat_id, user)
            if chat is None:
                return {"error": "Chat not found", "code": ERROR_UNKNOWN}
            # Use chat's project_id if project_id not explicitly provided
            if project_id is None:
                project_id = chat["project_id"]

        # If project_id is provided, verify it exists and belongs to the user
        if project_id is not None:
            user, _ = llm_get_user_name_and_model(auth)
            project = get_project(user, project_id)
            if project is None:
                return {"error": "Project not found", "code": ERROR_UNKNOWN}

        modelConfig = {
            "model": model,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_p": top_p,
            "chatter_type": auth_type,
            "openai_api_key": auth,
        }
        
        # Process connection args for RAG
        if useRAG:
            connection_args = ragConfig.get(ARGS_CONNECTION_ARGS, {})
            connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
            ragConfig[ARGS_CONNECTION_ARGS] = connection_args

        # Process connection args for KG
        if useKG:
            connection_args = kgConfig.get(ARGS_CONNECTION_ARGS, {})
            connection_args = process_connection_args(KG_VECTORSTORE, connection_args)
            kgConfig[ARGS_CONNECTION_ARGS] = connection_args

        if need_restrict_usage(auth_type, auth):
            return {"error": ERROR_EXCEEDS_TOKEN_LIMIT, "code": ERROR_EXCEEDS_TOKEN_LIMIT}

        # Use chat_id in session_id if provided
        effective_session_id = f"{sessionId}_{chat_id}" if chat_id is not None else sessionId

        (msg, usage, contexts) = chat(
            sessionId=effective_session_id,
            messages=messages,
            useRAG=useRAG,
            ragConfig=ragConfig,
            useKG=useKG,
            kgConfig=kgConfig,
            useAutoAgent=useAutoAgent,
            oncokbConfig=oncokbConfig,
            modelConfig=modelConfig,
            project_id=project_id,
        )
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": msg},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "contexts": contexts,
            "code": ERROR_OK,
        }
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED,
                "code": ERROR_MILVUS_CONNECT_FAILED,
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e)}

@app.post("/v1/rag/newdocument", description="creates new document")
def newDocument(
    authorization: Annotated[str | None, Header()],
    item: RagNewDocumentPostModel
):
    tmpFile = item.tmpFile
    filename = item.filename
    ragConfig = item.ragConfig
    if type(ragConfig) is str:
        ragConfig = json.loads(ragConfig)
    ragConfig[ARGS_CONNECTION_ARGS] = process_connection_args(
        RAG_VECTORSTORE, ragConfig[ARGS_CONNECTION_ARGS]
    )
    auth = llm_get_client_auth(authorization)
    embedding_func = llm_get_embedding_function(auth)
    # TODO: consider to be compatible with XinferenceDocumentEmbedder
    try:
        doc_id = new_embedder_document(
            tmp_file=tmpFile,
            filename=filename,
            rag_config=ragConfig,
            embedding_function=embedding_func
        )
        return {"id": doc_id, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED,
                "code": ERROR_MILVUS_CONNECT_FAILED,
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOWN}


@app.post("/v1/rag/alldocuments", description="retrieves all documents")
def getAllDocuments(
    authorization: Annotated[str | None, Header()],
    item: RagAllDocumentsPostModel,
):
    def post_process(docs: List[Any]):
        for doc in docs:
            doc["id"] = str(doc["id"])
        return docs

    auth = llm_get_client_auth(authorization)
    embedding_func = llm_get_embedding_function(auth)
    connection_args = item.connectionArgs
    connection_args = vars(connection_args)
    connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
    doc_ids = item.docIds
    try:
        docs = get_all_documents(
            connection_args=connection_args,
            doc_ids=doc_ids,
            embedding_function=embedding_func,
        )
        docs = post_process(docs)
        return {"documents": docs, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED,
                "code": ERROR_MILVUS_CONNECT_FAILED,
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOWN}


@app.delete("/v1/rag/document", description="removes a document")
def removeDocument(
    authorization: Annotated[str | None, Header()],
    item: RagDocumentDeleteModel,
):
    auth = llm_get_client_auth(authorization)
    embedding_func = llm_get_embedding_function(auth)
    docId = item.docId
    connection_args = item.connectionArgs
    connection_args = vars(connection_args)
    connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
    doc_ids = item.docIds
    if len(docId) == 0:
        return {"error": "Failed to find document"}
    try:
        remove_document(
            doc_id=docId,
            connection_args=connection_args,
            doc_ids=doc_ids,
            embedding_function=embedding_func,
        )
        return {"id": docId, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED,
                "code": ERROR_MILVUS_CONNECT_FAILED,
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOWN}


@app.post("/v1/rag/connectionstatus", description="returns connection status")
def getConnectionStatus(
    authorization: Annotated[str | None, Header()],
    item: RagConnectionStatusPostModel,
):
    try:
        auth = llm_get_client_auth(authorization)
        embedding_func = llm_get_embedding_function(auth)
        connection_args = item.connectionArgs
        connection_args = vars(connection_args)
        connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
        connected = get_vectorstore_connection_status(
            connection_args=connection_args,
            embedding_function=embedding_func,
        )
        return {
            "status": "connected" if connected else "disconnected",
            "code": ERROR_OK,
        }
    except MilvusException as e:
        return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOWN}


@app.post(
    "/v1/kg/connectionstatus", description="returns knowledge graph connection status"
)
def getKGConnectionStatus(
    item: KgConnectionStatusPostModel,
):
    try:
        connection_args = item.connectionArgs
        connection_args = vars(connection_args)
        connection_args = process_connection_args(RAG_KG, connection_args)
        connected = get_kg_connection_status(connection_args)
        return {
            "status": "connected" if connected else "disconnected",
            "code": ERROR_OK,
        }
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.post(
   "/v1/tokenusage", description="returns token usage for current user"
)
def getTokenUsage(
    authorization: Annotated[str | None, Header()],
    item: TokenUsagePostModel,
):
    try:
        auth = llm_get_client_auth(client_key=authorization)
        auth_type = llm_get_auth_type(auth)
        user, model = llm_get_user_name_and_model(auth, item.session_id, item.model)
        res = get_token_usage(user, model)
        res = res if res is not None else {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "model": model,
        }
        return {"code": ERROR_OK, "tokens": res, "auth_type": auth_type.value}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.post("/v1/projects/create", description="Create a new project")
def createProject(
    item: ProjectCreateModel,
):
    try:
        project = create_project("public", item.name, item.description)
        if project is None:
            return {"error": "Project already exists or creation failed", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK, "project": project}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.get("/v1/projects", description="Get all projects")
def getProjects():
    try:
        projects = get_user_projects("public")
        return {"code": ERROR_OK, "projects": projects}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.get("/v1/projects/{project_id}", description="Get a specific project")
def getProject(
    project_id: int,
):
    try:
        project = get_project("public", project_id)
        if project is None:
            return {"error": "Project not found", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK, "project": project}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.put("/v1/projects/{project_id}", description="Update a project")
def updateProject(
    project_id: int,
    item: ProjectUpdateModel,
):
    try:
        success = update_project("public", project_id, item.name, item.description)
        if not success:
            return {"error": "Project not found or update failed", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

@app.delete("/v1/projects/{project_id}", description="Delete a project")
def deleteProject(
    project_id: int,
):
    try:
        success = delete_project("public", project_id)
        if not success:
            return {"error": "Project not found or deletion failed", "code": ERROR_UNKNOWN}
        return {"code": ERROR_OK}
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ERROR_UNKNOWN}

if __name__ == "__main__":
    port: int = 5001
    uvicorn.run(app, host="0.0.0.0", port=port)
