import os
from unittest import TestCase
# from dotenv import load_dotenv
from unittest.mock import patch, MagicMock

from src.constants import (
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    AZURE_OPENAI_EMBEDDINGS_MODEL,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL,
)
from src.conversation_session import (
    ConversationSession,
    defaultModelConfig,
)
from src.datatypes import AuthTypeEnum

# load_dotenv()

class TestConversationSession(TestCase):
    def setUp(self):
        self.patch_os = patch.dict(os.environ, {
            OPENAI_API_TYPE: "azure",
            OPENAI_API_KEY: "abcdefg",
            OPENAI_DEPLOYMENT_NAME: "gpt4o-deploy-1",
            AZURE_OPENAI_ENDPOINT: "https://www.foo.com",
            OPENAI_API_VERSION: "2024-02-01",
            OPENAI_MODEL: "gpt-4",
            AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME: "embeddings",
            AZURE_OPENAI_EMBEDDINGS_MODEL: "embeddings",
        })
        self.patch_os.start()

        self.addCleanup(self.patch_os.stop)
        
        return super().setUp()
    
    def tearDown(self):
        return super().tearDown()

    @patch("src.conversation_session.AzureGptConversation")
    def test_ceate_conversation(self, mock_AzureGptConversation):
        mock_AzureGptConversation.return_value.find_rag_agent.return_value \
            = (None, None)
        mock_AzureGptConversation.return_value.get_last_injected_context.return_value \
            = None
        mock_AzureGptConversation.return_value.query.return_value \
            = ("Hello! How can I assist you today?", {
                'completion_tokens': 9, 'prompt_tokens': 8, 'total_tokens': 17
            }, None)

        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerAzureOpenAI"
        # modelConfig["openai_api_key"] = os.environ["TEST_OPENAI_API_KEY"]
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )
        
        (msg, usage, context) = session.chat(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            modelConfig=modelConfig,
        )
        self.assertEqual(msg, "Hello! How can I assist you today?")
        self.assertEqual(usage["total_tokens"], 17)


    @patch("src.conversation_session.GptConversation")
    def test_create_conversation_server(self, mock_GptConversation):
        mock_GptConversation.return_value.find_rag_agent.return_value \
            = (None, None)
        mock_GptConversation.return_value.get_last_injected_context.return_value \
            = None
        mock_GptConversation.return_value.query.return_value \
            = ("Hello! How can I assist you today?", {
                'completion_tokens': 9, 'prompt_tokens': 8, 'total_tokens': 17
            }, None)
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerOpenAI"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )
        (msg, usage, _) = session.chat(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            modelConfig=modelConfig,
        )
        self.assertEqual(msg, "Hello! How can I assist you today?")
        self.assertEqual(usage["total_tokens"], 17)

    @patch("src.conversation_session.AzureGptConversation")
    @patch("src.conversation_session.GptConversation")
    def test_validate_chatter_server_to_client(self, mock_GptConversation, mock_AzureGptConversation):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerAzureOpenAI"
        # modelConfig["openai_api_key"] = os.environ["TEST_OPENAI_API_KEY"]
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )

        mock_AzureGptConversation.assert_called_once()

        session._validate_chatter(
            modelConfig={
                "chatter_type": AuthTypeEnum.ClientOpenAI.value,
                "openai_api_key": "balahbalah",
            }
        )
        mock_GptConversation.assert_called_once()

    @patch("src.conversation_session.AzureGptConversation")
    @patch("src.conversation_session.GptConversation")
    def test_validate_chatter_client_to_server(self, mock_GptConversation, mock_AzureGptConversation):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ClientOpenAI"
        modelConfig["openai_api_key"] = "balahbalah"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )

        mock_GptConversation.assert_called_once()

        session._validate_chatter(
            modelConfig={
                "chatter_type": AuthTypeEnum.ServerAzureOpenAI.value,
            }
        )
        mock_AzureGptConversation.assert_called_once()

    @patch("src.conversation_session.AzureGptConversation")
    @patch("src.conversation_session.GptConversation")
    def test_validate_chatter_client_to_client(self, mock_GptConversation, mock_AzureGptConversation):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ClientOpenAI"
        modelConfig["openai_api_key"] = "balahbalah"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )

        mock_GptConversation.assert_called_once()

        session._validate_chatter(
            modelConfig={
                "chatter_type": AuthTypeEnum.ClientOpenAI.value,
                "openai_api_key": "foo",
            }
        )
        self.assertEqual(session.sessionData.modelConfig.chatter_type, AuthTypeEnum.ClientOpenAI)
        self.assertEqual(session.sessionData.modelConfig.openai_api_key, "foo")

    @patch("src.conversation_session.AzureGptConversation")
    @patch("src.conversation_session.GptConversation")
    def test_validate_chatter_client_change_model(self, mock_GptConversation, mock_AzureGptConversation):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ClientOpenAI"
        modelConfig["openai_api_key"] = "balahbalah"
        modelConfig["model"] = "gpt-3.5-turbo"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )

        mock_GptConversation.assert_called_once()

        session._validate_chatter(
            modelConfig={
                "chatter_type": AuthTypeEnum.ClientOpenAI.value,
                "model": "gpt-4o",
                "openai_api_key": "balahbalah",
            }
        )
        self.assertEqual(session.sessionData.modelConfig.chatter_type, AuthTypeEnum.ClientOpenAI)
        self.assertEqual(session.chatter.model_name, "gpt-4o")

class TestConversationSessionServerOpenAI(TestCase):
    def setUp(self):
        self.patch_os = patch.dict(os.environ, {
            OPENAI_API_KEY: "abcdefg",
        })
        self.patch_os.start()

        self.addCleanup(self.patch_os.stop)
        
        return super().setUp()
    
    def tearDown(self):
        return super().tearDown()
    
    @patch("src.conversation_session.AzureGptConversation")
    @patch("src.conversation_session.GptConversation")
    def test_validate_chatter_server_change_model(self, mock_GptConversation, mock_AzureGptConversation):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerOpenAI"
        modelConfig["model"] = "gpt-3.5-turbo"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )

        mock_GptConversation.assert_called_once()

        session._validate_chatter(
            modelConfig={
                "chatter_type": AuthTypeEnum.ServerOpenAI.value,
                "model": "gpt-4o",
            }
        )
        self.assertEqual(session.sessionData.modelConfig.chatter_type, AuthTypeEnum.ServerOpenAI)
        self.assertEqual(session.chatter.model_name, "gpt-4o")
