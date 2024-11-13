import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain Configuration 
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "Fin-agent"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Financial Modeling Prep Configuration
FINANCIAL_MODELING_PREP_API_KEY = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")

# Polygon Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# News API Configuration
news_client_id = os.getenv("news_client_id")
apha_api_key = os.getenv("apha_api_key")    


# Set environment variables
def setup_environment():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT 
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["FINANCIAL_MODELING_PREP_API_KEY"] = FINANCIAL_MODELING_PREP_API_KEY
    os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY
    os.environ["news_client_id"] = news_client_id
    os.environ["apha_api_key"] = apha_api_key