import datetime

from typing import Any, TypedDict, Dict, Annotated, Literal
from pydantic import BaseModel, Field
from operator import add
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI

import langgraph
from langgraph.graph import StateGraph, END
#from chains import get_finish_chain, get_supervisor_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage 

# Local imports
from utils.llm.llm import llm, llm_syn
from config.settings import setup_environment
from models.schemas import *
from config.members import get_team_members_details
from utils.chains import get_supervisor_chain
from models.personality import AgentPersonality
from utils.callback_tools import CustomConsoleCallbackHandler
#   Import agents
from agents.market import market_intelligence_agent
from agents.financial import financial_metrics_agent
from agents.sentiment import news_sentiment_agent
from agents.sql import sql_agent
# FinSage Agent Nodes

# Supervisor Node
def supervisor_node(state):
    """
    The supervisor node coordinates task delegation and validation.
    """
    print("\n" + "="*50)
    print("🎯 SUPERVISOR NODE")
    print(f"Current Input: {state['user_input']}")
    print(f"Analysis Date: {state['current_date']}")
    print(f"Personality in supervisor: {state.get('personality')}")
    
    # Add SQL data cutoff check
    sql_cutoff_date = datetime(2022, 12, 31)
    requires_historical = state['current_date'] > sql_cutoff_date
    print(f"Requires historical pre-2023 data consideration: {requires_historical}")

    chat_history = state.get("messages", [])
    supervisor_chain = get_supervisor_chain(llm, current_date=state['current_date'])
    print("="*50)
    # print("FULL CHAIN COMPONENTS:")
    # print(supervisor_chain)

    if not chat_history:
        chat_history.append(HumanMessage(state["user_input"]))
        print("Starting new conversation")
    
    # Debug the chain invocation
    print("\n=== Chain Invocation ===")
    print("Messages:", len(chat_history))
    print("Personality:", state.get("personality").get_prompt_context() if state.get("personality") else "None")
    
    output = supervisor_chain.invoke({
        "messages": chat_history,
        "personality": state.get("personality").get_prompt_context() if state.get("personality") else ""
    })
    print(f"\nNext Action: {output.next_action}")
    
    # Fix: Store complete task details in state
    state["current_task"] = {
        "description": output.task_description,
        "expected_output": output.expected_output,
        "validation_criteria": output.validation_criteria
    }
    
    #state["next_step"] = output.next_action


    if output.next_action == "FINISH" and len(chat_history) > 0:
        # state["next_step"] = "Reflection"
        # CHANGE THIS LATER!
        state["next_step"] = "Synthesizer"
    else:
        state["next_step"] = output.next_action
    
    state["messages"] = chat_history


    state["messages"] = chat_history
    
    print(f"\nNext Action: {output.next_action}")
    print(f"Task Description: {output.task_description}")
    print("="*50 + "\n")
    
    return state

# Synthesizer Node
def synthesize_responses(state):
    """
    Final node that synthesizes all agent responses into a comprehensive recommendation
    """
    state["callback"].write_agent_name("Investment Analysis Synthesis 🎯")
    print("\n" + "-"*50)
    print(" SYNTHESIS NODE")
    
    messages = state["messages"]
    print(f"Synthesizing {len(messages)} messages")
    
    messages = state["messages"]

    # These may bring some issues
    financial_metrics = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'FinancialMetrics'), '')
    news_sentiment = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'NewsSentiment'), '')
    market_intelligence = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'MarketIntelligence'), '')
    sql_data = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'SQLAgent'), '')
    
    synthesis_prompt = """You are an elite Wall Street analyst with 20+ years of experience, known for synthesizing complex financial data into actionable insights while maintaining strict objectivity. Your responses should be meticulous, data-driven, and tailored to each query type through careful step-by-step reasoning.
   
     "First - Carefully analyze the task by spelling it out loud.",
    "Then, break down the problem by thinking through it step by step and develop multiple strategies to solve the problem."
    "Then, examine the users intent and query develop a step by step plan to solve the problem.",
    "Work through your plan step-by-step, Using the SOURCE DATA and CONTEXT GIVEN.\n"

    CONTEXT:
    Analysis Date: {current_date}
    User Query: "{original_question}"
    
    INVESTMENT PROFILE:
    {personality}
    
    Analysis Guidelines Based on Profile:
    1. Risk Tolerance:
       - Conservative: Emphasize stability and risk mitigation
       - Moderate: Balance risk and opportunity
       - Aggressive: Focus on growth potential and higher returns
    
    2. Time Horizon:
       - Short-term: Prioritize immediate catalysts and technical factors
       - Medium-term: Balance current position with growth trajectory
       - Long-term: Focus on sustainable competitive advantages
    
    3. Investment Style:
       - Value: Emphasize intrinsic value and margin of safety
       - Growth: Focus on market opportunity and growth potential
       - Blend: Balance value and growth considerations


ANALYSIS FRAMEWORK:
    1. Query Classification & Scope:
    - Investment Decision (Buy/Sell/Hold)
    - Company Analysis
    - Market Trends
    - Risk Assessment
    - Performance Review
    - Industry Analysis
    - Competitive Analysis
    - Financial Health Check

    2. Data Integration & Cross-Validation:
    - Cross-reference all data sources for patterns and discrepancies
    - Weight recent data more heavily than historical
    - Consider market cycles and seasonality
    - Validate findings across multiple sources

    3. Profile-Based Analysis Guidelines:
    A. Risk Tolerance Alignment:
        - Conservative: Emphasize stability and risk mitigation
        - Moderate: Balance risk and opportunity
        - Aggressive: Focus on growth potential and higher returns
    
    B. Time Horizon Consideration:
        - Short-term: Prioritize immediate catalysts and technical factors
        - Medium-term: Balance current position with growth trajectory
        - Long-term: Focus on sustainable competitive advantages
    
    C. Investment Style Integration:
        - Value: Emphasize intrinsic value and margin of safety
        - Growth: Focus on market opportunity and growth potential
        - Blend: Balance value and growth considerations

    4. Comprehensive Analysis Structure:
    A. Executive Summary:
        - Clear position/recommendation
        - Confidence level (High/Medium/Low)
        - Key decision drivers
    
    B. Detailed Analysis:
        - Financial Health (Financial Metrics)
        - Market Position (Market Intelligence)
        - Sentiment Analysis (News)
        - Historical Context (SQL data)
    
    C. Risk Assessment:
        - Market risks
        - Company-specific risks
        - Macro factors
    
    D. Action Plan:
        - Specific entry/exit points
        - Position sizing recommendations
        - Risk management guidelines

    SOURCE DATA:
    - Financial Metrics: {financial_metrics}
    - Market Intelligence: {market_intelligence}
    - News & Sentiment: {news_sentiment}
    - Historical Data: {sql_data}

   RULES TO ALWAYS FOLLOW:
   1. ANY DATA YOU USE MUST BE FROM THE SOURCE DATA PROVIDED
   ALL DATA FROM FINANCIAL_METRICS, MARKET_INTELLIGENCE, MUST BE shown  and should be in tabular format.
   2. ALL NUMERICAL DATA YOU SHOULD PROVIDE IN BEAUTIFUL TABLE FORMAT. FOR EXAMPLE:
    |---------|---------|---------|
    | Value1  | Value2  | Value3  |
   3. ALL TEXT DATA YOU SHOULD PROVIDE IN MARKDOWN FORMAT.
   4. ALL TABLES MUST BE PROPERLY FORMATTED WITH ALIGNED COLUMNS
   5. ANY SOURCE DATA YOU USE LIKE NEWS AND OTHER INSIGHTS MUST INCLUDE URLS FROM NEWS_SENTIMENT SOURCE DATA
   6. MUST FOLLOW Response Structure and Response Framework to draft your final response as per user query 
    
    NOTE : please consider the investment profile of user like risk tolerance, time horizon and investment style to draft your final response as per user query 
    1. User Query Classification:
        First, classify the query type:
        - Investment Decision (Buy/Sell/Hold)
        - Company Analysis
        - Market Trends
        - Risk Assessment
        - Performance Review
        - Industry Analysis
        - Competitive Analysis
        - Financial Health Check
        - Other (Adapt accordingly)
    
    2. RESPONSE FRAMEWORK:
        Output in the following format:
        ALL numerical data you should provide in beautiful table format. for example:
        |---------|---------|---------|
        | Value1  | Value2  | Value3  |
        ALL text data you should provide in markdown format.
        ALL tables must be properly formatted with aligned columns
        Any source data you use like news and other insights must include urls from news_sentiment source data:
        - Direct URL to source
        - Publication date
        - Credibility rating
        - Summary of the article
        - Key points of the article


    3. Response Structure:
        Based on user query type and investment profile, provide:
        - Direct answer to the main question in a good format depending on the query type adjust the format accordingly like a big trading firm analyst give detailed answers when needed
        - Supporting quantitative evidence like tables, charts, or specific data points to support the answer never miss tables from  market_intelligence source like insider Transactions and others data data and finacial_metrics and all URLs from news_sentiment source data
        - Relevant context and comparisons to help the user understand the answer
        - Specific actionable insights
        - Time-sensitive considerations
        - for buying or selling or holding give a  primary recommendation along confidence levels(High, Medium, Low) supported by data and analysis.
        - for portfolio management give a recommendation along with a plan to manage the portfolio and confidence levels(High, Medium, Low) supported and percentage allocation by data and analysis

    4. Data Integration:
        - Prioritize data points most relevant to the query
        - Cross-reference different data sources
        - Highlight agreements/disagreements in data
        - Explain significant patterns or anomalies
        - Focus on query-specific metrics

    5. Critical Analysis:
        - Challenge assumptions
        - Consider alternative viewpoints
        - Identify potential biases
        - Note data limitations
        - Provide confidence levels

    6. Actionable Conclusions:
        - Clear, specific recommendations like a seasoned analyst supported by data and analysis and investment profile and urls from news_sentiment source data
        - Next steps or monitoring points 
        - Time-sensitive factors
        - Risk considerations
        - Success metrics
    
    QUALITY STANDARDS:
    - Every claim must be supported by specific data and investment profile and all URLs from news_sentiment source data
    - ALL numerical data you should provide in beautiful table format. for example:
    |---------|---------|---------|
    | Value1  | Value2  | Value3  |
    - All analyses must be relevant to the original query and investment profile
    - Time periods must be clearly stated
    - Uncertainties must be acknowledged
    - Recommendations must be actionable

    PRESENTATION GUIDELINES:
    1. All tables must be properly formatted with aligned columns
    2. Use consistent decimal places for numerical data
    3. Include units where applicable (%, $, etc.)
    4. Highlight critical values or signals
    5. Include timestamps for all data points
    6. Provide source citations for all data especially news and other insights should be backed by urls from news_sentiment source data
    7. Use markdown formatting for text and tables
    8. Ensure all data is up-to-date and relevant to the query
    9. Use bold or italic formatting for emphasis when appropriate
    10. Use bullet points or numbered lists for structured information

    Remember:
    - Stay focused on answering the specific query 
    - Adapt detail level to query complexity 
    - Align all recommendations with user's investment profile
    - Prioritize relevance over comprehensiveness
    - Be explicit about confidence levels
    - Include forward-looking implications when appropriate"""
    
    personality = state.get("personality")
    messages = [
        SystemMessage(content=synthesis_prompt.format(
            current_date=state.get("current_date", "Not specified"),
            original_question=state["user_input"],
            personality=personality.get_prompt_context() if personality else "",
            financial_metrics=financial_metrics,
            market_intelligence=market_intelligence,
            news_sentiment=news_sentiment,
            sql_data=sql_data
        )),
        HumanMessage(content="Synthesize the analyses into a focused response that directly addresses the query in a best format supported by evidence and data(SHOULD BE IN TABLE FORMAT for all numerical data) and investment profile and urls from news_sentiment source data")
    ]
    print(messages)
    
    final_response = llm_syn.invoke(messages)
    state["callback"].on_tool_end(final_response.content)
    state["messages"].append(AIMessage(content=final_response.content, name="FinalSynthesis"))
    return state


# Build the graph
def define_graph():
    """
    Defines and returns a graph representing the financial analysis workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("FinancialMetricsAgent", financial_metrics_agent)
    workflow.add_node("NewsSentimentAgent", news_sentiment_agent)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("MarketIntelligenceAgent", market_intelligence_agent)
    
    workflow.add_node("SQLAgent", sql_agent)
    
    workflow.add_node("Synthesizer", synthesize_responses)
     # Add Reflection node with retry policy
    # workflow.add_node(
    #     "Reflection", 
    #     #reflection_node
    #     # retry=RetryPolicy(
    #     #     max_attempts=2,
    #     #     retry_on=lambda x: isinstance(x, ValueError) and "Analysis incomplete" in str(x)
    #     # )
    # )
    # Set entry point
    workflow.set_entry_point("Supervisor")
    
    # Define available agents
    members = [
        "FinancialMetricsAgent",
        "NewsSentimentAgent",
        "MarketIntelligenceAgent",
        "SQLAgent"
    ]
    
    # Add edges from agents to Supervisor
    for member in members:
        workflow.add_edge(member, "Supervisor")
    
    # Add edge from Reflection back to Supervisor
    #workflow.add_edge("Reflection", "Supervisor")
    
    # Add conditional routing from Supervisor
    conditional_map = {k: k for k in members}
    #conditional_map["Reflection"] = "Reflection"
    conditional_map["Synthesizer"] = "Synthesizer"
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next_step"],
        conditional_map
    )
    
    # Add edge from Synthesizer to END
    workflow.add_edge("Synthesizer", END)
    
    return workflow.compile()

FinSage_agent = define_graph()

def __main__():
    """
    Main function to build and run the market intelligence agent graph.
    """
    # Initialize environment
    setup_environment()
    
    # Personality - default
    sample_personality = AgentPersonality()
    # Set up graph
    graph = define_graph()
    # callback handler
    callback_handler = CustomConsoleCallbackHandler()
    # Initialize state
    state = {
        "current_date": datetime.now(),
        "user_input": "SHould I buy stocks from apple?",
        "messages": [],
        "next_step": "",
        "config": {},
        "callback": callback_handler,
        "personality": sample_personality,
        "news_sentiment_agent_internal_state": {
            "agent_executor_tools": {},
            "full_response": {},
            "all_tools_eval": {"passed": [], "stats": []},
            "topic_adherence_eval": {"passed": [], "reason": []}
        },
        "financial_metrics_agent_internal_state": {
            "agent_executor_tools": {},
            "full_response": {},
            "all_tools_eval": {"passed": [], "stats": []},
            "topic_adherence_eval": {"passed": [], "reason": []}
        },
        "market_intelligence_agent_internal_state": {
            "agent_executor_tools": {},
            "full_response": {},
            "all_tools_eval": {"passed": [], "stats": []},
            "topic_adherence_eval": {"passed": [], "reason": []}
        },
        "sql_agent_internal_state": {
            "messages" : [], 
            "agent_executor_tools": {},
            "full_response": {},
            "all_tools_eval": {"passed": [], "stats": []},
            "topic_adherence_eval": {"passed": [], "reason": []}
        }
    }

    # Run the graph
    result = graph.invoke(state)
    return result

if __name__ == "__main__":
    __main__()