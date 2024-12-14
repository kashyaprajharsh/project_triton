from dotenv import load_dotenv



# SQL AGENT
from typing import Annotated, Dict, List, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel
from operator import add

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage 
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import langgraph
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.message import AnyMessage, add_messages



# local modules
from FinSage.utils.llm.llm import llm
from FinSage.models.schemas import *
from FinSage.utils.callback_tools import CustomConsoleCallbackHandler
from FinSage.prompts.system_prompts import SQL_AGENT_QUERY_PROMPT, SQL_AGENT_ANALYZE_PROMPT

# Load environment variables
load_dotenv()


# from sql_agent.py (modified):
db_loc = "stock_db.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_loc}")
database_schema = db.get_table_info()

# Create SQL toolkit and tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
tools_names = [tool.name for tool in tools]

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# Latest date in the db
db_latest_date = "2022-09-30" 

# ########## HELPER FUNCTIONS ############### #
def clean_sql_query(query: str) -> str:
    """Remove markdown formatting and clean the SQL query"""
    # Remove markdown SQL markers
    query = query.replace("```sql", "").replace("```", "")
    # Remove any leading/trailing whitespace
    query = query.strip()
    return query


# ########## NODES ############### #
def analyze_question(state: AgentState) -> dict:
    """Analyze the question to determine relevant tables"""
    try:
        messages = state["messages"]
        question = state["user_input"]
        
        # Get task details from supervisor
        task = state.get("current_task", {})
        print(task)
        
        state["sql_agent_internal_state"]["agent_tools"] = tools_names
        
        tables = list_tables_tool.invoke("")
        schema = db.get_table_info()
        # Include task details in the analysis prompt
        analysis_prompt = SQL_AGENT_ANALYZE_PROMPT.format(
            question=question,
            schema=schema,
            tables=tables,
            db_latest_date = db_latest_date,
            task_description=task.get("description", ""),
            expected_output=task.get("expected_output", ""),
            validation_criteria=task.get("validation_criteria", [])
        )
        
    
        
        messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=question)
        ]
        sllm = llm.with_structured_output(AnalyzedQuestion)
        analysis = sllm.invoke(messages)
        
        state['sql_agent_internal_state']['date_available'] = analysis.date_available
        state['sql_agent_internal_state']['relevant_tables'] = analysis.relevant_tables
        print("Set date_available to:", state['sql_agent_internal_state']['date_available'])
        print('Set state[\'relevant_tables\'][\'tables\'] to: ' , state['sql_agent_internal_state']['relevant_tables']['tables'], ' type: ', type(state['sql_agent_internal_state']['relevant_tables']['tables']))

        state['messages'] = state["messages"] + [AIMessage(content=str(analysis.response))]
        
        return state
    except Exception as e:
        print(f"Error in analyze_question: {str(e)}")
        
        state['messages'] = state["messages"] + [AIMessage(content=f"Error in analysis: {str(e)}")]
        return state

def get_schemas(state: AgentState) -> dict:
    """Get schemas for relevant tables"""
    try:
        messages = state["messages"]
        tables = messages[-1].content.split(",")
        schemas = []
        for table in tables:
            table = table.strip()
            schema = get_schema_tool.invoke(table)
            schemas.append(schema)
        
        return {
            "messages": state["messages"] + [AIMessage(content="\n".join(schemas))]
        }
    except Exception as e:
        print(f"Error in get_schemas: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error getting schemas: {str(e)}")]
        }

def generate_query(state: AgentState) -> dict:
    """Generate SQL query based on schemas and question"""
    try:
        question = state["user_input"]   # state["messages"][0].content
        schemas = state["messages"][-1].content
        
        messages = [
            SystemMessage(content=SQL_AGENT_QUERY_PROMPT.format(schema=schemas)),
            HumanMessage(content=question)
        ]
        
        query = llm.invoke(messages)
        # Clean the query before returning
        cleaned_query = clean_sql_query(query.content)
        
        return {
            "messages": state["messages"] + [AIMessage(content=cleaned_query)]
        }
    except Exception as e:
        print(f"Error in generate_query: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error generating query: {str(e)}")]
        }

def validate_query(state: AgentState) -> dict:
    """Validate and potentially correct the SQL query"""
    try:
        query =  state["messages"][-1].content
        schema = db.get_table_info()
        
        messages = [
            SystemMessage(content="""Validate this SQL query and return ONLY the corrected query with NO additional text or explanation:
            {query}

            Schema:
            {schema}

            Check for:
            - Proper table joins
            - Correct column names
            - Appropriate WHERE clauses
            - Proper data type handling
            - SQL injection prevention
            - DO NOT modify any ticker symbols in WHERE clauses""".format(query=query, schema=schema)), HumanMessage(content=query)
        ]
        
        validated = llm.invoke(messages)
        # Clean the validated query before returning
        cleaned_query = clean_sql_query(validated.content)
        
        # Verify the ticker hasn't been changed
        if "WHERE" in query and "WHERE" in cleaned_query:
            original_ticker = query.split("WHERE")[1].strip().split("=")[1].strip()
            validated_ticker = cleaned_query.split("WHERE")[1].strip().split("=")[1].strip()
            if original_ticker != validated_ticker:
                return {
                    "messages": state["messages"] + [AIMessage(content=query)]  # Return original query
                }
        
        return {
            "messages": state["messages"] + [AIMessage(content=cleaned_query)]
        }
    except Exception as e:
        print(f"Error in validate_query: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error validating query: {str(e)}")]
        }

def execute_query(state: AgentState) -> dict:
    """Execute the SQL query"""
    try:
        query = state["messages"][-1].content
        # Make sure the query is clean before execution
        clean_query = clean_sql_query(query)
        result = db.run(clean_query)
        return {
            "messages": state["messages"] + [AIMessage(content=str(result))]
        }
    except Exception as e:
        print(f"Error in execute_query: {str(e)}")
        
        # Add to state the details about the wrong query 
        state["sql_agent_internal_state"]["wrong_generated_queries"].append({"Wrong query": clean_query, "Error message": f"Error executing query: {str(e)}"})
        state["messages"] = state["messages"] + [AIMessage(content=f"Error executing query: {str(e)}")]
        return state

def format_results(state: AgentState) -> dict:
    """Format the query results into a readable response"""
    print(" INSIDE FORMAT RESULTS")
    try:
        # Find the SQL query from previous messages
        sql_query = None
        for message in state["messages"]:
            if isinstance(message, AIMessage) and "SELECT" in message.content:
                sql_query = message.content
                break
        
        result = state["messages"][-1].content
        print("THIS IS THE RESULT: ", result)
        if result.startswith("Error:"):
            state["sql_agent_internal_state"]["wrong_formatted_results"].append({"Formatted content": result, "Error message": result})
            return state
            
            #return state # {"messages": state["messages"]}
        ##################
        messages = [
            SystemMessage(content="""Format these SQL results into a clear, readable response.
            Include both the SQL query used and the results in your response.
            Format as:
            SQL Query:
            <query>
            
            Results:
            <formatted results>"""),
            HumanMessage(content=f"Query: {sql_query}\nResults: {result}")
        ]
        
        formatted = llm.invoke(messages)
        state["messages"] = state["messages"] + [AIMessage(content=formatted.content)]
        return state
    except Exception as e:
        print(f"Error in format_results: {str(e)}")
        state["sql_agent_internal_state"]["wrong_formatted_results"].append({"Formatted content": formatted.content, "Error message": f"Error in format_results: {str(e)}"})
        state["messages"] = state["messages"] + [AIMessage(content=f"Error formatting results: {str(e)}")]
        return state

# ########## CONDITIONAL EDGES ############### #
def check_date_availability_and_tables(state: AgentState) -> Literal["get_schemas", "end"]:
    """ Check whether the date enter by the user if after the latest date available in the Database"""
    print("DATE AVAILABILITY: ", state["sql_agent_internal_state"]["date_available"])
    print(" RELEVANT TABLES FOUND : ", state['sql_agent_internal_state']['relevant_tables']['tables'])

    date_unavailable = state["sql_agent_internal_state"]["date_available"].lower() == "false"
    no_relevant_tables = len(state['sql_agent_internal_state']['relevant_tables']['tables']) == 0
    
    if date_unavailable or no_relevant_tables:
        return "end"
    return "get_schemas"

def should_continue(state: AgentState) -> Literal["retry_schema", "end"]:
    """Decide whether to retry with schema (retry_schema), or end the workflow (end)"""
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else ""
    
    if "Error: no such table" in content and len(state["sql_agent_internal_state"]["wrong_formatted_results"]) < 3:
        return "retry_schema"
  
    return "end"

def generate_again(state: AgentState) -> Literal["generate_again" , "format_results"]:
    """Decide whether to generate the query again (generte_again), or go to format_results (format_results)"""
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else ""
    if "Error" in content and len(state["sql_agent_internal_state"]["wrong_generated_queries"]) < 3:
        return "generate_again"
    else: 
        return "format_results"

# ########## COMPILE GRAPH ############### #

def define_graph():
    # Create and configure the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("get_schemas", get_schemas)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("format_results", format_results)

    # Add edges
    workflow.add_edge(START, "analyze_question")
    workflow.add_edge("get_schemas", "generate_query")
    workflow.add_edge("generate_query", "validate_query")
    workflow.add_edge("validate_query", "execute_query")

    # Add conditional edges
    workflow.add_conditional_edges("analyze_question" , check_date_availability_and_tables, {
        "get_schemas" : "get_schemas",
        "end": END
    })

    workflow.add_conditional_edges(
        "execute_query", 
        generate_again,
        {
            "format_results" : "format_results",
            "generate_again" : "generate_query"
        }

    )
    workflow.add_conditional_edges(
        "format_results",
        should_continue,
        {
            "retry_schema": "get_schemas",
            "end": END
        }
    )

    # Compile the workflow
    return workflow.compile()

sql_agent = define_graph()

def __main__():
    """
    Main function to build and run the market intelligence agent graph.
    """
    from config.settings import setup_environment
    from models.personality import AgentPersonality
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
        "user_input": "Analyze the market condition for Tesla (TSLA) in 2020",
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
            "agent_tools": [],   # just the tool names
            "date_available": "",
            "relevant_tables": {
                "tables": [],
                "explanation": ""
            },
            "response": "",  # Add this field to match schema
            "wrong_generated_queries": [],
            "wrong_formatted_results": []
        }
    }

    # Run the graph
    result = graph.invoke(state)
    return result

if __name__ == "__main__":
    __main__()