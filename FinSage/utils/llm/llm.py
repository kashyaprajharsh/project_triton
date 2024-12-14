from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import setup_environment

setup_environment()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
llm_syn = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=0.0,max_output_tokens=8192)
#llm_syn = ChatGoogleGenerativeAI(model="gemini-exp-1206",temperature=0.0,max_output_tokens=8192)
