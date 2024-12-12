from datetime import datetime

from typing import Any, TypedDict, Dict, Annotated, Literal
from pydantic import BaseModel, Field
from operator import add
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import langgraph
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage 

# Local imports
from llms import llm
from config import setup_environment
from personality import AgentPersonality
from tools import news_sentiment_tools
from prompts import get_news_sentiment_agent_prompt, NEWS_SENTIMENT_TOPIC_ADHERENCE_PROMPT
from custom_callback_handler import CustomConsoleCallbackHandler
from schemas import *

# ##### HELPER FUNCTIONS #########
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, max_iterations: int = 2,  return_intermediate_steps: bool = True) -> AgentExecutor:
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
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations = max_iterations, return_intermediate_steps = return_intermediate_steps, verbose = True)
    return executor

def get_tools_call_eval_stats(result: Dict):
    """Helper function to format the output and store evaluation stats"""
    print("\n📊 TOOL EVALUATION SUMMARY")
    print("=" * 50)
    
    # Create stats dictionary to store in state
    run_stats = {
        "timestamp": datetime.now(),
        "all_tools_used": result["all_tools_used"],
        "tool_counts": result['tool_usage']['call_counts'],
        "errors": {
            "invalid_tools": result['tool_usage']['errors']['invalid_tools'],
            "execution_errors": result['tool_usage']['errors']['execution_errors'],
            "parser_errors": result['tool_usage']['errors']['parser_errors']
        }
    }
    
    # Print evaluation results
    all_tools_status = "✅" if result["all_tools_used"] else "❌"
    print(f"\n🎯 Overall Status:")
    print(f"  • All Required Tools Used: {all_tools_status}")
    
    print(f"\n📋 Tool Inventory:")
    print(f"  • Available Tools: {', '.join(result['tool_usage']['available_tools'])}")
    print(f"  • Successfully Used: {', '.join(result['tool_usage']['used_tools'])}")
    print(f"  • Not Used: {', '.join(result['tool_usage']['unused_tools'])}")
    
    print("\n📈 Usage Statistics:")
    for tool, count in result['tool_usage']['call_counts'].items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {tool}: {count} calls")
    
    # Error Summary
    has_errors = any(len(errs) > 0 for errs in result['tool_usage']['errors'].values())
    if has_errors:
        print("\n⚠️ Error Summary:")
        errors = result['tool_usage']['errors']
        
        if errors['invalid_tools']:
            print("\n  Invalid Tool Attempts:")
            for err in errors['invalid_tools']:
                print(f"  • Requested: {err['requested']}")
                print(f"    Available: {', '.join(err['available'])}")
        
        if errors['execution_errors']:
            print("\n  Tool Execution Errors:")
            for err in errors['execution_errors']:
                print(f"  • Tool: {err['tool']}")
                print(f"    Input: {err['input']}")
                print(f"    Error: {err['error']}")
        
        if errors['parser_errors']:
            print("\n  Parser Errors:")
            for err in errors['parser_errors']:
                print(f"  • Input: {err['input']}")
                print(f"    Error: {err['error']}")
    
    print("\n🔍 Detailed Tool Execution Log:")
    for step in result["tools_used"]:
        status_emoji = {
            "success": "✅",
            "parser_error": "🔍",
            "invalid_tool": "❌",
            "execution_error": "⚠️"
        }.get(step['status'], "❓")
        
        print(f"\n  {status_emoji} Tool: {step['tool']}")
        print(f"    Status: {step['status']}")
        print(f"    Input: {step['input']}")
        if step['status'] == "success":
            print(f"    Output: {str(step['output'])[:100]}...")  # Truncate long outputs
    
    print("\n" + "=" * 50)
    
    return run_stats


# News Sentiment Agent Nodes
def news_sentiment_node(state):
    """
    Handles news analysis and sentiment tracking using tools from tools.py
    """
    
    sentiment_agent = create_agent(
        llm,
        news_sentiment_tools,
        get_news_sentiment_agent_prompt(state.get("current_date"), state.get("personality"), state.get("user_input"))
    )
    
    state["callback"].write_agent_name("News & Sentiment Agent 📰")
    output = sentiment_agent.invoke(
        {"messages": state["messages"]}, # state["messages"] # [HumanMessage(state["user_input"])]
        {"callbacks": [state["callback"]], } , return_intermediate_steps = True
    )
    print(f"Analysis complete - Output length: {len(output.get('output', ''))}")
    
    state["messages"].append(
        AIMessage(content=output.get("output"), name="NewsSentiment") # changed it from Human to AIMessage
    )

    # ADDED: sentiment_agent tools:
    available_tools = {tool.name: 0 for tool in sentiment_agent.tools}                                           
    state["news_sentiment_agent_internal_state"]["agent_executor_tools"] = available_tools
    state["news_sentiment_agent_internal_state"]["full_response"] = output # output contains all the messages
    #print("THIS IS THE OUTPUT: ", output)

    print("-"*50 + "\n")
    return state

# Evaluate all tools called:
def evaluate_all_tools_called(state):
    """Evaluates tool usage and stores statistics in state"""
    sample_response = state['news_sentiment_agent_internal_state']
    
    # This dictionary will be used for later evaluation statistics
    result = {
        "answer": sample_response['full_response']["output"],
        "tools_used": [],
        "all_tools_used": False,
        "tool_usage": {
            "available_tools": list(sample_response['agent_executor_tools'].keys()),
            "used_tools": set(),
            "unused_tools": set(),
            "call_counts": sample_response['agent_executor_tools'].copy(),
            "errors": {
                "invalid_tools": [],
                "execution_errors": [],
                "parser_errors": []
            }
        }
    }

    # Process intermediate steps
    for action, observation in sample_response['full_response']["intermediate_steps"]:
        # print( 'ACTION: ', action)
        # print(' OBSERVATION: ', observation)
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Determine status based on observation
        status = "success"  # Default, but will be immediately evaluated
        
        if tool_name == "_Exception":
            status = "parser_error"
            result["tool_usage"]["errors"]["parser_errors"].append({
                "input": tool_input,
                "error": observation
            })
        elif isinstance(observation, dict) and "requested_tool_name" in observation:
            status = "invalid_tool"
            result["tool_usage"]["errors"]["invalid_tools"].append({
                "requested": tool_name,
                "available": observation.get("available_tool_names", [])
            })
        elif isinstance(observation, dict) and next(iter(observation), "").lower().startswith("error"):
            
            status = "execution_error"
            result["tool_usage"]["errors"]["execution_errors"].append({
                "tool": tool_name,
                "input": tool_input,
                "error": observation
            })

        tool_result = {
            "tool": tool_name,
            "input": tool_input,
            "output": observation,
            "status": status
        }
        
        # Only count successful executions in usage statistics
        if status == "success":
            result["tool_usage"]["used_tools"].add(tool_name)
            result["tool_usage"]["call_counts"][tool_name] += 1

        result["tools_used"].append(tool_result)

    # Calculate unused tools
    
    # Set of tool names (strings) that were available but not used
    result["tool_usage"]["unused_tools"] = set(result["tool_usage"]["available_tools"]) - result["tool_usage"]["used_tools"]
    result["all_tools_used"] = len(result["tool_usage"]["unused_tools"]) == 0

    # Store evaluation stats in state
    run_stats = get_tools_call_eval_stats(result)
    
    # Initialize all_tools_eval if needed
    if 'stats' not in state['news_sentiment_agent_internal_state']['all_tools_eval']:
        state['news_sentiment_agent_internal_state']['all_tools_eval'] = {
            'stats': [],
            'passed': []
        }
    
    # Append new stats and pass/fail status
    state['news_sentiment_agent_internal_state']['all_tools_eval']['stats'].append(run_stats)
    state['news_sentiment_agent_internal_state']['all_tools_eval']['passed'].append(result["all_tools_used"])
    
    return state

# Evaluate topic adherene
def evaluate_topic_adherence(state):
    print(' INSIDE evaluate_topic_adherence')
    messages = [
        SystemMessage(content=NEWS_SENTIMENT_TOPIC_ADHERENCE_PROMPT.format(
            question=state['user_input'],
            answer= state['news_sentiment_agent_internal_state']['full_response']['output']
        ))
    ]
    llm_evaluator = llm.with_structured_output(LLM_TopicAdherenceEval)
    response = llm_evaluator.invoke(messages)
    
    # Append to the internal state:
    state['news_sentiment_agent_internal_state']['topic_adherence_eval']['passed'].append(response.passed)
    state['news_sentiment_agent_internal_state']['topic_adherence_eval']['reason'].append(response.reason)
    return state

# News Sentiment Agent Conditional Edges

# Conditional edge to decide wether to go to topic adherence or retry tool calling
def execute_again_all_tools_called(state):
    print("INSIDE execute_again_all_tools_called")
    # all_tools_called_eval_passed will contain a booleean
    passed = state['news_sentiment_agent_internal_state']['all_tools_eval']['passed'][-1]
    iterations = len(state['news_sentiment_agent_internal_state']['all_tools_eval']['passed'])
    print("passed value:" ,passed )
    print('iterations: ', iterations, 'values: ' , state['news_sentiment_agent_internal_state']['all_tools_eval']['passed'])

    if passed or iterations >= 2:
   
        return "EvaluateTopicAdherence"
    else:
        print('GO BACK TO THE AGENT, tools not passed')
        return "NewsSentimentAgent"

# Conditional edge to decide wether to END of go back to NewsSentimentAgent
def execute_again_topic_adherence(state):
    print('INSIDE execute_again_topic_adherence')
    
    # Check if 'topic_adherence_eval' has any evaluations
    if not state['news_sentiment_agent_internal_state']['topic_adherence_eval']['passed']:
        print("No topic adherence evaluations found.")
        return "NewsSentimentAgent"  
    
    # Access the latest evaluation
    last_passed = state['news_sentiment_agent_internal_state']['topic_adherence_eval']['passed'][-1].lower()
    # Check how many evaluations occured
    iterations = len(state['news_sentiment_agent_internal_state']['topic_adherence_eval']['passed'])
    
    print("TOPIC ADHERENCE EVALUATION PASSED:", last_passed)
    print("NUMBER OF ITERATIONS FOR TOPIC ADHERENCE:", iterations)

    if last_passed == "true" or iterations >= 2: 
        print(f'ENDING! iterations {iterations}, value of topic_adherence: {last_passed}')
        return "end"
    else:
        print(f'RETURN TO AGENT, adherence failed! iterations {iterations}, value of topic_adherence: {last_passed}')
        return "NewsSentimentAgent"

# Define the graph
def define_graph():
    """
    Defines and returns a graph representing the financial analysis workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("NewsSentimentAgent", news_sentiment_node)
    workflow.add_node("EvaluateAllToolsCalled", evaluate_all_tools_called)
    workflow.add_node("EvaluateTopicAdherence", evaluate_topic_adherence)
    
    # Set entry point
    workflow.set_entry_point("NewsSentimentAgent")
    
    # Add edges
    workflow.add_edge("NewsSentimentAgent", "EvaluateAllToolsCalled")
    
    workflow.add_conditional_edges("EvaluateAllToolsCalled", execute_again_all_tools_called, 
    {
        "EvaluateTopicAdherence": "EvaluateTopicAdherence", #"EvaluateTopicAdherence",
        "NewsSentimentAgent": "NewsSentimentAgent"
    }
      )
    
    workflow.add_conditional_edges("EvaluateTopicAdherence", execute_again_topic_adherence,                           
    {
        "end": END, 
        "NewsSentimentAgent": "NewsSentimentAgent"
    }
      )
                            
    return workflow.compile()


news_sentiment_agent = define_graph()

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
        "user_input": "How will Trump's 2024 election impact the New Yor Stock Exchange",
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
        }
    }

    # Run the graph
    result = graph.invoke(state)
    return result

if __name__ == "__main__":
    __main__()