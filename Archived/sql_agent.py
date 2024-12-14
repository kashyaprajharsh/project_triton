from typing import Annotated, Literal, Dict, List, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from config import setup_environment
from llms import llm


setup_environment()


db_loc = "stock_db.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_loc}")
database_schema = db.get_table_info()





# Create SQL toolkit and tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define system prompts
ANALYZE_PROMPT = """You are a financial data SQL expert. Analyze the user's question to determine which tables contain relevant information.

Available tables: {tables}
Schema information: {schema}

Guidelines:
1. For financial metrics, always check both 'fundamentals' and 'price' tables
2. For company analysis, consider historical data patterns
3. Map common company names to tickers:
   - Apple/AAPL
   - Microsoft/MSFT
   - Google/GOOGL
   - Amazon/AMZN
   - Tesla/TSLA

Return ONLY the relevant table names separated by commas, nothing else."""

QUERY_PROMPT = """You are a SQL expert. Use the following schema to help answer questions:
{schema}

Given an input question, create a syntactically correct SQLite query that joins the relevant tables.
IMPORTANT: 
- Return ONLY the SQL query without any markdown formatting or explanation
- Only query for relevant columns, not all columns
- Current date is 2022-09-30
- When company names are mentioned, map them to their correct ticker symbols:
  * Microsoft or microsoft -> MSFT
  * Apple or apple -> AAPL
  * Google or google -> GOOGL
  * etc.
- Always include appropriate WHERE clauses to filter for the specific company mentioned

DO NOT:
- Use markdown formatting (no ```sql or ``` tags)
- Make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
- Include any explanations before or after the query
- Default to AAPL or any other company if a specific company is mentioned
"""

# VALIDATION_PROMPT = """Validate this SQL query for common mistakes:
# {query}

# Schema:
# {schema}

# Check for:
# - Proper table joins
# - Correct column names
# - Appropriate WHERE clauses
# - Proper data type handling
# - SQL injection prevention

# Return ONLY the corrected query if needed, or the original query if valid.
# DO NOT include any markdown formatting or explanations."""

def clean_sql_query(query: str) -> str:
    """Remove markdown formatting and clean the SQL query"""
    # Remove markdown SQL markers
    query = query.replace("```sql", "").replace("```", "")
    # Remove any leading/trailing whitespace
    query = query.strip()
    return query



# Define node functions
def analyze_question(state: State) -> dict:
    """Analyze the question to determine relevant tables"""
    try:
        messages = state["messages"]
        question = messages[-1].content if messages else ""
        
        tables = list_tables_tool.invoke("")
        schema = db.get_table_info()
        
        messages = [
            SystemMessage(content=ANALYZE_PROMPT.format(tables=tables, schema=schema)),
            HumanMessage(content=question)
        ]
        
        analysis = llm.invoke(messages)
        
        return {
            "messages": state["messages"] + [AIMessage(content=str(analysis.content))]
        }
    except Exception as e:
        print(f"Error in analyze_question: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error in analysis: {str(e)}")]
        }

def get_schemas(state: State) -> dict:
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

def generate_query(state: State) -> dict:
    """Generate SQL query based on schemas and question"""
    try:
        question = state["messages"][0].content
        schemas = state["messages"][-1].content
        
        messages = [
            SystemMessage(content=QUERY_PROMPT.format(schema=schemas)),
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

def validate_query(state: State) -> dict:
    """Validate and potentially correct the SQL query"""
    try:
        query = state["messages"][-1].content
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
- DO NOT modify any ticker symbols in WHERE clauses""".format(query=query, schema=schema)),
            HumanMessage(content=query)
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

def execute_query(state: State) -> dict:
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
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error executing query: {str(e)}")]
        }

def format_results(state: State) -> dict:
    """Format the query results into a readable response"""
    try:
        # Find the SQL query from previous messages
        sql_query = None
        for message in state["messages"]:
            if isinstance(message, AIMessage) and "SELECT" in message.content:
                sql_query = message.content
                break
        
        result = state["messages"][-1].content
        if result.startswith("Error:"):
            return {"messages": state["messages"]}
        
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
        return {
            "messages": state["messages"] + [AIMessage(content=formatted.content)]
        }
    except Exception as e:
        print(f"Error in format_results: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error formatting results: {str(e)}")]
        }
# Define the conditional edge function
def should_continue(state: State) -> Literal["continue", "retry_schema", "end"]:
    """Decide whether to continue, retry with schema, or end the workflow"""
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else ""
    
    if "Error: no such table" in content:
        return "retry_schema"
    elif "Error" in content:
        return "continue"
    return "end"

# Create and configure the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("analyze_question", analyze_question)
workflow.add_node("get_schemas", get_schemas)
workflow.add_node("generate_query", generate_query)
workflow.add_node("validate_query", validate_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("format_results", format_results)

# Add edges
workflow.add_edge(START, "analyze_question")
workflow.add_edge("analyze_question", "get_schemas")
workflow.add_edge("get_schemas", "generate_query")
workflow.add_edge("generate_query", "validate_query")
workflow.add_edge("validate_query", "execute_query")
workflow.add_edge("execute_query", "format_results")

# Add conditional edges
workflow.add_conditional_edges(
    "format_results",
    should_continue,
    {
        "continue": "generate_query",
        "retry_schema": "get_schemas",
        "end": END
    }
)

# Compile the workflow
agent = workflow.compile()

# Create a wrapper function for easier use
def query_database(question: str):
    """
    Query the database using the agent workflow
    
    Args:
        question (str): The question to ask about the data
        
    Returns:
        str: The formatted response or error message
    """
    try:
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }
        response = agent.invoke(initial_state)
        
        # Extract the final response
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            final_message = messages[-1].content if messages else "No response generated"
            return final_message
        return response
    except Exception as e:
        print(f"Error querying database: {str(e)}")
        return f"Error: {str(e)}"

# Example usage


# # Example usage
# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod

# display(
#     Image(
#         agent.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )

if __name__ == "__main__":
    # Test the agent
    questions = [
        "What are all the company stock values present in the database. provide me the unique companies (ticker) in the price table?",
        "What is the stock price of apple - ticker AAPL for today?",
        "Which is the industry pe value for apple?",
        "Provide me the financial information of microsoft like revenue profit and EBIDTA for today",
        "What are the fundamentals of apple - ticker AAPL for today?",
        "What is the stock price of microsoft - ticker MSFT for today?",
        "Which is the industry pe value for microsoft?",
        
    ]
    
    for question in questions:
        print("\nQuestion:", question)
        print("-" * 50)
        result = query_database(question)
        print("Response:", result)
        print("=" * 80)