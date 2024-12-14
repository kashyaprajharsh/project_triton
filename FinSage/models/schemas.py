from typing import Literal, TypedDict,  Optional, List, Union, Dict , Any, Annotated, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from operator import add
from pydantic import BaseModel, Field
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
# local imports
from models.personality import AgentPersonality

# __________________________________________________________________________________________ #
# ___________________________________ Routing Schema _______________________________________ #
# __________________________________________________________________________________________ #
class RouteSchema(BaseModel):
    next_action: Literal[
        "FinancialMetricsAgent",
        "NewsSentimentAgent",
        "MarketIntelligenceAgent",
        "SQLAgent",
        "Reflection",
        "Synthesizer",
        "FINISH"
    ]= Field(
        ...,
        title="Next",
        description="Select the next role",
    )
    task_description: str = Field(
        description="Detailed description of what the agent should analyze"
    )
    expected_output: str = Field(
        description="Description of expected deliverables"
    )
    validation_criteria: List[str] = Field(
        description="List of specific points to validate in the agent's response"
    )

# __________________________________________________________________________________________ #
# __________________________ Pydantic Structures for Agent Evaluation ______________________ #
# __________________________________________________________________________________________ #
class TopicAdherenceEval(BaseModel):
    passed: Annotated[list[Any] , add] # will hold values of True or False
    reason: Annotated[list[Any] , add] # will hold reasons for the True or False

# Add state for all_tools_called_eval
class AllToolsEval(BaseModel):
    passed: Annotated[list[Any], add]  # bool value for if evalution passed = all tools were called successfully
    stats: Annotated[list[Any] , add]  # stats for each tool call, errors etc..
    
# Pydantic structure for model to evaluate response:
class LLM_TopicAdherenceEval(BaseModel):
    passed: str
    reason: str

# Pydantic for Analyzed Question by SQL Agent
class AnalyzedQuestion(BaseModel):
    relevant_tables: Dict[str, Any] = {
        "tables": [],  # List of relevant tables
        "explanation": ""  # Explanation for table selection or why no tables are relevant
    }
    response: str  # Will contain table names or error message
    date_available: str  # Default to "true" if no date mentioned

# __________________________________________________________________________________________ #
# _________________________________ AGENT'S INTERNAL STATE _________________________________ #
# __________________________________________________________________________________________ #

# Sentiment News Agent
class SentimentNewsState(BaseModel):
    agent_executor_tools: dict 
    full_response: dict
    all_tools_eval: AllToolsEval                    
    topic_adherence_eval: TopicAdherenceEval        

# Financial Metrics Agent
class FinancialMetricsState(BaseModel):
    agent_executor_tools: dict 
    full_response: dict
    all_tools_eval: AllToolsEval                   
    topic_adherence_eval: TopicAdherenceEval        

# Market Intelligence Agent
class MarketIntelligenceState(BaseModel):
    agent_executor_tools: dict 
    full_response: dict
    all_tools_eval: AllToolsEval                    # Annotated[list[Any], add]
    topic_adherence_eval: TopicAdherenceEval        # Annotated[list[TopicAdherenceEval] , add]

# SQL Agent
class SQLAgentState(BaseModel): 
    agent_tools: List[Any] # list of tools available to the SQLAgent
    date_available: str
    relevant_tables: Dict[str, Any] = {
        "tables": [],  # List of relevant tables
        "explanation": ""  # Explanation for table selection or why no tables are relevant
    }
    response: str  # Will contain table names or error message
    wrong_generated_queries : Annotated[List[Dict[str, Any]], add] # This will track details about wrong queries generated in the generate_query node
    wrong_formatted_results : Annotated[List[Dict[str, Any]], add] 


# Overall Agent state
class AgentState(TypedDict):
    current_date: datetime
    user_input: str
    messages: list[BaseMessage]
    next_step: str
    config: dict
    callback: Any
    personality: AgentPersonality
    news_sentiment_agent_internal_state: SentimentNewsState
    financial_metrics_agent_internal_state: FinancialMetricsState
    market_intelligence_agent_internal_state: MarketIntelligenceState
    sql_agent_internal_state: SQLAgentState
    current_task: dict

