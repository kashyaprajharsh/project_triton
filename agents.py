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
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def supervisor_node(state):
    """
    The supervisor node is the main node in the graph. It is responsible for routing to the correct agent.
    """
    chat_history = state.get("messages", [])
    supervisor_chain = get_supervisor_chain(llm)
    if not chat_history:
        chat_history.append(HumanMessage(state["user_input"]))
    output = supervisor_chain.invoke({"messages": chat_history})
    state["next_step"] = output.next_action
    state["messages"] = chat_history
    return state


def financial_metrics_node(state):
    """
    Handles fundamental analysis and financial metrics using tools from tools.py
    """
    metrics_agent = create_agent(
        llm,
        financial_metrics_tools,
        get_financial_metrics_agent_prompt()
    )
    
    state["callback"].write_agent_name("Financial Metrics Agent 📊")
    output = metrics_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="FinancialMetrics")
    )
    return state

def news_sentiment_node(state):
    """
    Handles news analysis and sentiment tracking using tools from tools.py
    """
    sentiment_agent = create_agent(
        llm,
        news_sentiment_tools,
        get_news_sentiment_agent_prompt()
    )
    
    state["callback"].write_agent_name("News & Sentiment Agent 📰")
    output = sentiment_agent.invoke(
        {"messages": state["messages"]},
        {"callbacks": [state["callback"]]}
    )
    
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="NewsSentiment")
    )
    return state
def market_intelligence_node(state):
    """
    Handles market data analysis using tools from tools.py
    """
    market_agent = create_agent(
        llm,
        market_intelligence_tools,
        get_market_intelligence_agent_prompt()
    )
    
    state["callback"].write_agent_name("Market Intelligence Agent 📈")
    output = market_agent.invoke(
        {"messages": state["messages"]},
        {"callbacks": [state["callback"]]}
    )
    
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="MarketIntelligence")
    )
    return state

def sql_agent_node(state):
    """
    Handles SQL database queries using the LangGraph agent from sql_agent.py
    """
    state["callback"].write_agent_name("SQL Database Agent 🗄️")
    question = state["messages"][-1].content
    
    try:
        # Use the query_database function from sql_agent.py
        result = query_database(question)
        
        # Format the response in a more readable way
        formatted_result = f"Database Query Results:\n{result}"
        
        # Show the result through the callback handler
        state["callback"].on_tool_end(formatted_result)
        
        state["messages"].append(
            AIMessage(content=formatted_result, name="SQLAgent")
        )
    except Exception as e:
        error_message = f"Error executing database query: {str(e)}"
        state["callback"].on_tool_error(error_message)
        state["messages"].append(
            AIMessage(content=error_message, name="SQLAgent")
        )
    
    return state

def synthesize_responses(state):
    """
    Final node that synthesizes all agent responses into a comprehensive recommendation
    """
    state["callback"].write_agent_name("Investment Analysis Synthesis 🎯")
    
    messages = state["messages"]
    financial_metrics = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'FinancialMetrics'), '')
    news_sentiment = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'NewsSentiment'), '')
    market_intelligence = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'MarketIntelligence'), '')
    sql_data = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'SQLAgent'), '')
    
    synthesis_prompt = """Based on the comprehensive analysis provided, synthesize a detailed investment recommendation:

    Context:
    Original Query: "{original_question}"

    Analysis Components:
    1. Financial Analysis:
    {financial_metrics}

    2. Market Intelligence:
    {market_intelligence}

    3. News & Sentiment:
    {news_sentiment}

    4. Historical Context:
    {sql_data}

    Required Output Sections:
    1. Executive Summary
       - Key Findings
       - Primary Recommendation
       - Confidence Level

    2. Investment Thesis
       - Fundamental Drivers
       - Technical Factors
       - Catalyst Timeline
       - Risk/Reward Profile

    3. Risk Analysis
       - Market Risks
       - Company-Specific Risks
       - External Factors
       - Mitigation Strategies

    4. Action Plan
       - Specific Recommendations
       - Entry/Exit Points
       - Position Sizing
       - Monitoring Metrics

    5. Timeline & Milestones
       - Implementation Schedule
       - Review Points
       - Adjustment Triggers
       - Performance Metrics
    """
    
    messages = [
        SystemMessage(content=synthesis_prompt.format(
            original_question=state["user_input"],
            financial_metrics=financial_metrics,
            market_intelligence=market_intelligence,
            news_sentiment=news_sentiment,
            sql_data=sql_data
        )),
        HumanMessage(content="Synthesize all analyses into a comprehensive investment recommendation.")
    ]
    
    final_response = llm.invoke(messages)
    state["callback"].on_tool_end(final_response.content)
    state["messages"].append(AIMessage(content=final_response.content, name="FinalSynthesis"))
    return state



def reflection_node(state):
    """
    Reviews and critiques the outputs from other agents to improve analysis quality.
    Uses the ReflexionFramework to evaluate responses and suggest improvements.
    """
    
    state["callback"].write_agent_name("Reflection Agent 🤔")
    
    # Gather all previous analyses
    messages = state["messages"]
    
    # Get the original query
    original_query = state["user_input"]
    
    # Get outputs from each agent
    agent_outputs = {
        "FinancialMetrics": next((msg.content for msg in messages if getattr(msg, 'name', '') == 'FinancialMetrics'), ''),
        "NewsSentiment": next((msg.content for msg in messages if getattr(msg, 'name', '') == 'NewsSentiment'), ''),
        "MarketIntelligence": next((msg.content for msg in messages if getattr(msg, 'name', '') == 'MarketIntelligence'), ''),
        "SQLAgent": next((msg.content for msg in messages if getattr(msg, 'name', '') == 'SQLAgent'), '')
    }
    
    reflection_prompt = """As a critical financial analyst, evaluate the collective analyses and identify areas for improvement:

    Original Query: {query}

    Agent Outputs:
    {outputs}

    Evaluate across these dimensions:
    1. Completeness
       - Are all relevant aspects covered?
       - What key metrics or analyses are missing?
       - Which areas need deeper investigation?

    2. Consistency & Integration
       - Are there conflicts between different analyses?
       - How well do the analyses complement each other?
       - What additional connections could be made?

    3. Evidence & Reasoning
       - Is each conclusion well-supported?
       - Are there potential biases or assumptions?
       - What additional evidence would strengthen the analysis?

    4. Actionability
       - How clear and specific are the recommendations?
       - Are risk factors adequately addressed?
       - Is the timeline and implementation guidance practical?

    Provide specific improvement suggestions for each dimension.
    """
    
    messages = [
        SystemMessage(content=reflection_prompt.format(
            query=original_query,
            outputs="\n\n".join([f"{k}:\n{v}" for k,v in agent_outputs.items()])
        )),
        HumanMessage(content="Evaluate the analyses and provide improvement suggestions.")
    ]
    
    reflection = llm.invoke(messages)
    state["callback"].on_tool_end(reflection.content)
    state["messages"].append(AIMessage(content=reflection.content, name="Reflection"))
    return state



class AgentState(TypedDict):
    user_input: str
    messages: list[BaseMessage]
    next_step: str
    config: dict
    callback: Any



def define_graph():
    """
    Defines and returns a graph representing the financial analysis workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("FinancialMetricsAgent", financial_metrics_node)
    workflow.add_node("NewsSentimentAgent", news_sentiment_node)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("MarketIntelligenceAgent", market_intelligence_node)
    workflow.add_node("SQLAgent", sql_agent_node)
    workflow.add_node("Reflection", reflection_node)
    workflow.add_node("Synthesizer", synthesize_responses)
    
    # Set entry point
    workflow.set_entry_point("Supervisor")
    
    # Define available agents
    members = [
        "FinancialMetricsAgent",
        "NewsSentimentAgent",
        "MarketIntelligenceAgent",
        "SQLAgent"
    ]
    
    # Add edges from agents to Reflection
    for member in members:
        workflow.add_edge(member, "Supervisor")
        
    # Add conditional routing from Supervisor
    conditional_map = {k: k for k in members}
    conditional_map["Reflection"] = "Reflection"
    conditional_map["Synthesizer"] = "Synthesizer"
    
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next_step"],
        conditional_map
    )
    
    # Add edge from Reflection to Synthesizer
    workflow.add_edge("Reflection", "Synthesizer")
    workflow.add_edge("Synthesizer", END)
    
    return workflow.compile()

