from typing import Any, TypedDict
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from chains import get_finish_chain, get_supervisor_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage 

from tools import financial_metrics_tools, news_sentiment_tools, market_intelligence_tools
from prompts import get_financial_metrics_agent_prompt, get_news_sentiment_agent_prompt, get_market_intelligence_agent_prompt
from sql_agent import query_database
from datetime import datetime
from llms import llm

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """
    Creates an agent using the specified ChatOpenAI model, tools, and system prompt.

    Args:
        llm : LLM to be used to create the agent.
        tools (list): The list of tools to be given to the worker node.
        system_prompt (str): The system prompt to be used in the agent.

    Returns:
        AgentExecutor: The executor for the created agent.
    """
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True
    )
    return executor



def news_sentiment_node(user_query: str):
    """
    Handles news analysis and sentiment tracking using tools from tools.py
    """
    print("\n" + "-"*50)
    print("📰 NEWS SENTIMENT NODE")
    
    analysis_date = datetime(2024, 11, 18)
    personality = "Personality in supervisor: risk_tolerance=<RiskTolerance.AGGRESSIVE: 'aggressive'> time_horizon=<TimeHorizon.SHORT_TERM: 'short_term'> investment_style=<InvestmentStyle.BLEND: 'blend'>"
    
    # Make the query more specific and structured to avoid repetitive tool calls
    enhanced_query = f"""Analyze the following query in a single comprehensive pass:
    
    Query: {user_query}
    
    Instructions:
    1. Use each tool only once
    2. Combine all findings into a single coherent response
    3. Focus on the most relevant and recent information
    4. If you have sufficient information, provide your analysis without making additional tool calls"""
    
    sentiment_agent = create_agent(
        llm,
        news_sentiment_tools,
        get_news_sentiment_agent_prompt(current_date=analysis_date, personality=personality)
    )
    
    # Add max_iterations parameter to limit tool calls
    executor = AgentExecutor(
        agent=sentiment_agent.agent,
        tools=news_sentiment_tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=3  # Limit the number of tool calls
    )
    
    output = executor.invoke(
        {"messages": [HumanMessage(content=enhanced_query)]},
    )
    #Print the raw tool message
    for step in output.get("intermediate_steps", []):
        action = step[0]  
        response = step[1]  
        print(f"\nToolMessage(action='{action.tool}', input='{action.tool_input}', content='{response}')")
            
        print(f"\nAnalysis complete - Output length: {len(output.get('output', ''))}")
        print(output.get("output"))
    print(output)
    return output.get("output")

# def financial_metrics_node(user_query: str):
#     """
#     Handles fundamental analysis and financial metrics using tools from tools.py

#     Args:
#         user_query (str): The user's question or request for financial analysis
#     """
#     print("\n" + "-"*50)
#     print("📊 FINANCIAL METRICS NODE")
    
#     metrics_agent = create_agent(
#         llm,
#         financial_metrics_tools,
#         get_financial_metrics_agent_prompt()
#     )
    
#     messages = [HumanMessage(content=user_query)]
#     output = metrics_agent.invoke({"messages": messages})
    
#     # Print the raw tool message
#     for step in output.get("intermediate_steps", []):
#         action = step[0]  
#         response = step[1]  
#         print(f"\nToolMessage(action='{action.tool}', input='{action.tool_input}', content='{response}')")
        
#     print(f"\nAnalysis complete - Output length: {len(output.get('output', ''))}")
#     print(output.get("output"))
#     print(output)
#     return output.get("output")

result = news_sentiment_node("Is The Nvidia a profitable company?")
# result = financial_metrics_node("What is the P/E ratio of AAPL?")
print(result)