
from biochatter.vectorstore import DocumentEmbedder, DocumentReader
from typing import Any, Dict, List, Optional
import logging

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from src.constants import (
    ARGS_CHUNK_SIZE,
    ARGS_CONNECTION_ARGS,
    ARGS_OVERLAP_SIZE,
    ARGS_RESULT_NUM,
    ARGS_SPLIT_BY_CHAR
)

logger = logging.getLogger(__name__)

def new_embedder_document(
        tmp_file: str, 
        filename: str, 
        rag_config: Any,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None,
    ):
    rag_agent = DocumentEmbedder(
      used=True,
      chunk_size=rag_config[ARGS_CHUNK_SIZE],
      chunk_overlap=rag_config[ARGS_OVERLAP_SIZE],
      split_by_characters=rag_config[ARGS_SPLIT_BY_CHAR],
      n_results=rag_config[ARGS_RESULT_NUM],
      embeddings=embedding_function,
      connection_args=rag_config[ARGS_CONNECTION_ARGS]
    )
    rag_agent.connect()
    reader = DocumentReader()
    docs = reader.load_document(tmp_file)
    if len(docs) > 0:
        for doc in docs:
            doc.metadata.update({"source": filename})
    logger.info('save_document')
    return rag_agent.save_document(docs)

def get_all_documents(
        connection_args: Dict, 
        doc_ids: Optional[List[str]] = None,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None=None,
    ):
    rag_agent = DocumentEmbedder(
        connection_args=connection_args,
        documentids_workspace=doc_ids,
        embeddings=embedding_function
    )
    rag_agent.connect()
    return rag_agent.get_all_documents()

def remove_document(
        doc_id: str,
        connection_args, 
        doc_ids: Optional[List[str]] = None,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None=None,
    ):
    rag_agent = DocumentEmbedder(
        connection_args=connection_args,
        documentids_workspace=doc_ids,
        embeddings=embedding_function,
    )
    rag_agent.connect()
    rag_agent.remove_document(doc_id=doc_id)

def get_connection_status(
        connection_args: Optional[Dict]=None,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None=None,
    ) -> bool:
    if connection_args is None:
        return False
    try:
        rag_agent = DocumentEmbedder(
            connection_args=connection_args,
            embeddings=embedding_function,
        )
        rag_agent.connect()
        return True
    except Exception as e:
        return False
