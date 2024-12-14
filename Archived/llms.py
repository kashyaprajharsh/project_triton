from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import setup_environment

setup_environment()



llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
