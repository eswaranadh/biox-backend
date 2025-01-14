
from unittest import TestCase
from dotenv import load_dotenv

from src.conversation_session import (
    ConversationSession,
    defaultModelConfig,
)

load_dotenv()

class TestConversationSession(TestCase):
    def setUp(self):
        return super().setUp()
    
    def tearDown(self):
        return super().tearDown()
    
    def test_ceate_conversation(self):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerAzureOpenAI"
        # modelConfig["openai_api_key"] = os.environ["TEST_OPENAI_API_KEY"]
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )
        
        res = session.chat(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            modelConfig=modelConfig,
        )
        self.assertIsNot(res, None)

    def test_create_conversation_server(self):
        modelConfig = {**defaultModelConfig}
        modelConfig["chatter_type"] = "ServerAzureOpenAI"
        session = ConversationSession(
            "abcdefg",
            modelConfig
        )
        res = session.chat(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            modelConfig=modelConfig,
        )
        self.assertIsNot(res, None)

