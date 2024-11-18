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
from prompts import get_financial_metrics_agent_prompt, get_news_sentiment_agent_prompt, get_market_intelligence_agent_prompt, get_reflection_prompt
from sql_agent import query_database
from datetime import datetime
from llms import llm
from datetime import datetime
from personality import AgentPersonality

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, current_date: datetime = None):
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
        verbose=True
    )
    return executor


def supervisor_node(state):
    """
    The supervisor node is the main node in the graph. It is responsible for routing to the correct agent.
    """
    print("\n" + "="*50)
    print("🎯 SUPERVISOR NODE")
    print(f"Current Input: {state['user_input']}")
    print(f"Analysis Date: {state['current_date']}")
    print(f"Personality in supervisor: {state.get('personality')}")

    chat_history = state.get("messages", [])
    supervisor_chain = get_supervisor_chain(llm, current_date=state['current_date'])
    
    if not chat_history:
        chat_history.append(HumanMessage(state["user_input"]))
        print("Starting new conversation")
    
    # Pass personality to the chain invocation
    output = supervisor_chain.invoke({
        "messages": chat_history,
        "personality": state.get("personality").get_prompt_context() if state.get("personality") else ""
    })
    print(f"Next Action: {output.next_action}")
    
    if output.next_action == "FINISH" and len(chat_history) > 0:
        state["next_step"] = "Reflection"
    else:
        state["next_step"] = output.next_action
    
    state["messages"] = chat_history
    print("="*50 + "\n")
    return state


def financial_metrics_node(state):
    """
    Handles fundamental analysis and financial metrics using tools from tools.py
    """
    print("\n" + "-"*50)
    print("📊 FINANCIAL METRICS NODE")
    
    metrics_agent = create_agent(
        llm,
        financial_metrics_tools,
        get_financial_metrics_agent_prompt(state.get("current_date"), state.get("personality"))
    )
    
    state["callback"].write_agent_name("Financial Metrics Agent 📊")
    output = metrics_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    print(f"Analysis complete - Output length: {len(output.get('output', ''))}")
    
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="FinancialMetrics")
    )
    print("-"*50 + "\n")
    return state

def news_sentiment_node(state):
    """
    Handles news analysis and sentiment tracking using tools from tools.py
    """
    print("\n" + "-"*50)
    print("📰 NEWS SENTIMENT NODE")
    
    sentiment_agent = create_agent(
        llm,
        news_sentiment_tools,
        get_news_sentiment_agent_prompt(state.get("current_date"), state.get("personality"))
    )
    
    state["callback"].write_agent_name("News & Sentiment Agent 📰")
    output = sentiment_agent.invoke(
        {"messages": state["messages"]},
        {"callbacks": [state["callback"]]}
    )
    print(f"Analysis complete - Output length: {len(output.get('output', ''))}")
    
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="NewsSentiment")
    )
    print("-"*50 + "\n")
    return state
def market_intelligence_node(state):
    """
    Handles market data analysis using tools from tools.py
    """
    print("\n" + "-"*50)
    print("📈 MARKET INTELLIGENCE NODE")
    
    market_agent = create_agent(
        llm,
        market_intelligence_tools,
        get_market_intelligence_agent_prompt(state.get("current_date"), state.get("personality"))
    )
    
    state["callback"].write_agent_name("Market Intelligence Agent 📈")
    output = market_agent.invoke(
        {"messages": state["messages"]},
        {"callbacks": [state["callback"]]}
    )
    print(f"Analysis complete - Output length: {len(output.get('output', ''))}")
    
    state["messages"].append(
        HumanMessage(content=output.get("output"), name="MarketIntelligence")
    )
    print("-"*50 + "\n")
    return state

def sql_agent_node(state):
    """
    Handles SQL database queries using the LangGraph agent from sql_agent.py
    """
    print("\n" + "-"*50)
    print("🗄️ SQL AGENT NODE")
    question = state["messages"][-1].content
    print(f"Processing query: {question}")
    
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
    
    print("-"*50 + "\n")
    return state

def synthesize_responses(state):
    """
    Final node that synthesizes all agent responses into a comprehensive recommendation
    """
    state["callback"].write_agent_name("Investment Analysis Synthesis 🎯")
    print("\n" + "-"*50)
    print("🎯 SYNTHESIS NODE")
    
    messages = state["messages"]
    print(f"Synthesizing {len(messages)} messages")
    
    messages = state["messages"]
    financial_metrics = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'FinancialMetrics'), '')
    news_sentiment = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'NewsSentiment'), '')
    market_intelligence = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'MarketIntelligence'), '')
    sql_data = next((msg.content for msg in messages if getattr(msg, 'name', '') == 'SQLAgent'), '')
    
    synthesis_prompt = """You are an expert financial advisor synthesizing insights from multiple specialized analyses. Your responses should be data-driven and tailored to each query type.

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

    SOURCE DATA:
    - Financial Metrics: {financial_metrics}
    - Market Intelligence: {market_intelligence}
    - News & Sentiment: {news_sentiment}
    - Historical Data: {sql_data}

    RESPONSE FRAMEWORK:
    NOTE : please consider the investment profile of user  like risk tolerance, time horizon and investment style to draft your final response as per user query 
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

    2. Response Structure:
        Based on user query type and investment profile, provide:
        - Direct answer to the main question in a good format depending on the query type adjust the format accordingly like a big trading firm analyst give detailed answers when needed
        - Supporting quantitative evidence like tables, charts, or specific data points to support the answer
        - Relevant context and comparisons to help the user understand the answer
        - Specific actionable insights
        - Time-sensitive considerations
        - for buying or selling or holding give a  primary recommendation along confidence levels(High, Medium, Low) supported by data and analysis.
        - for portfolio management give a recommendation along with a plan to manage the portfolio and confidence levels(High, Medium, Low) supported and percentage allocation by data and analysis

    3. Data Integration:
        - Prioritize data points most relevant to the query
        - Cross-reference different data sources
        - Highlight agreements/disagreements in data
        - Explain significant patterns or anomalies
        - Focus on query-specific metrics

    4. Critical Analysis:
        - Challenge assumptions
        - Consider alternative viewpoints
        - Identify potential biases
        - Note data limitations
        - Provide confidence levels

    5. Actionable Conclusions:
        - Clear, specific recommendations like a seasoned analyst supported by data and analysis
        - Next steps or monitoring points 
        - Time-sensitive factors
        - Risk considerations
        - Success metrics
    
    QUALITY STANDARDS:
    - Every claim must be supported by specific data and investment profile
    - All analyses must be relevant to the original query and investment profile
    - Time periods must be clearly stated
    - Uncertainties must be acknowledged
    - Recommendations must be actionable

    Remember:
    - Stay focused on answering the specific query 
    - Adapt detail level to query complexity 
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
        HumanMessage(content="Synthesize the analyses into a focused response that directly addresses the query in a best format supported by evidence and data.")
    ]
    print(messages)
    
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
    print("\n" + "-"*50)
    print("🤔 REFLECTION NODE")
    
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
    print("Available analyses:", agent_outputs)

    # Get reflection prompt and format with query and outputs
    reflection_template = get_reflection_prompt(state.get("current_date"))
    formatted_prompt = reflection_template.format(
        query=original_query,
        outputs="\n\n".join([f"{k}:\n{v}" for k,v in agent_outputs.items()])
    )
    
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content="Evaluate the analyses and provide improvement suggestions.")
    ]
    
    reflection = llm.invoke(messages)
    state["callback"].on_tool_end(reflection.content)
    state["messages"].append(AIMessage(content=reflection.content, name="Reflection"))
    print("Reflection complete")
    print("-"*50 + "\n")
    return state



class AgentState(TypedDict):
    current_date: datetime
    user_input: str
    messages: list[BaseMessage]
    next_step: str
    config: dict
    callback: Any
    personality: AgentPersonality



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
    
    # Add edges from agents to Supervisor
    for member in members:
        workflow.add_edge(member, "Supervisor")
    
    # Add edge from Reflection back to Supervisor
    workflow.add_edge("Reflection", "Supervisor")
    
    # Add conditional routing from Supervisor
    conditional_map = {k: k for k in members}
    conditional_map["Reflection"] = "Reflection"
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