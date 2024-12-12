import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
#from custom_callback_handler import CustomStreamlitCallbackHandler
#from agents import define_graph
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from config import setup_environment
from personality import AgentPersonality, RiskTolerance, TimeHorizon, InvestmentStyle

# Local Imports
from custom_callback_handler import CustomStreamlitCallbackHandler
from finsage import FinSage_agent
from plotting_tools import *

setup_environment()

# Debug helper function
def debug_state(state):
    """Debug helper to print state contents"""
    print("\n=== DEBUG: State Contents ===")
    for key, value in state.items():
        if key == "messages":
            print(f"messages: {len(value)} messages")
        elif key == "personality":
            print(f"personality: {value.get_prompt_context()}")
        else:
            print(f"{key}: {value}")
    print("===========================\n")


# Page configuration        
st.set_page_config(
    page_title="FinSage AI | Intelligent Financial Analysis",
    page_icon="https://img.icons8.com/?size=100&id=YagodtnP71eo&format=png&color=000000",
    layout="wide",
    initial_sidebar_state="expanded"
)


# App Header with Logo and Title
col1, col2 = st.columns([0.2, 4])
with col1:
    st.image("https://img.icons8.com/?size=100&id=YagodtnP71eo&format=png&color=000000", width=80)
with col2:
    st.title("FinSage AI" )
    st.markdown("*Your Intelligent Financial Analysis Assistant*")

print("\n=== DEBUG: Initializing Application ===")

# Initialize chat history
if "messages" not in st.session_state:
    print("Initializing new chat history")
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to FinSage AI! I'm your advanced financial analysis assistant. How can I help you make data-driven financial decisions today?"}
    ]

# Create the agent flow
#flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()
print("Agent flow graph initialized")

# Enhanced message display
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
        st.write(message["content"])
    print(f"Displayed message from {message['role']}")

def process_agent_output(output, response_container):
    """Process and display the agent output in a structured way"""
    print("\n=== DEBUG: Processing Agent Output ===")
    try:
        messages = output.get("messages", [])
        final_synthesis = None
        
        for message in messages:
            if hasattr(message, 'name') and message.name == "FinalSynthesis":
                final_synthesis = message.content
                print("Found final synthesis")
                break
        
        if final_synthesis:
            print("Displaying final synthesis in UI")
            st.markdown(final_synthesis)
                
    except Exception as e:
        print(f"Error in process_agent_output: {str(e)}")
        st.error(f"Error processing output: {str(e)}")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("## 📈 Investment Profile")
    st.markdown("---")
    
    print("\n=== DEBUG: Setting up Investment Profile ===")
    
    # Risk Tolerance with visual indicator
    risk_tolerance = st.selectbox(
        "🎯 Risk Tolerance",
        options=[rt.value for rt in RiskTolerance],
        index=[rt.value for rt in RiskTolerance].index(RiskTolerance.MODERATE.value),
        help="Choose your risk tolerance level"
    )
    print(f"Selected Risk Tolerance: {risk_tolerance}")
    
    # Time Horizon with visual indicator
    time_horizon = st.selectbox(
        "⏳ Time Horizon",
        options=[th.value for th in TimeHorizon],
        index=[th.value for th in TimeHorizon].index(TimeHorizon.MEDIUM_TERM.value),
        help="Select your investment time horizon"
    )
    print(f"Selected Time Horizon: {time_horizon}")
    
    # Investment Style with visual indicator
    investment_style = st.selectbox(
        "💼 Investment Style",
        options=[style.value for style in InvestmentStyle],
        index=[style.value for style in InvestmentStyle].index(InvestmentStyle.BLEND.value),
        help="Choose your preferred investment style"
    )
    print(f"Selected Investment Style: {investment_style}")
    
    # Initialize or update personality
    personality = AgentPersonality(
        risk_tolerance=RiskTolerance(risk_tolerance),
        time_horizon=TimeHorizon(time_horizon),
        investment_style=InvestmentStyle(investment_style)
    )
    
    st.session_state.personality = personality
    
    # Debug print
    print("\n=== DEBUG: Current Investment Profile ===")
    print(f"Risk Tolerance: {risk_tolerance}")
    print(f"Time Horizon: {time_horizon}")
    print(f"Investment Style: {investment_style}")
    print(f"Personality Context: {st.session_state.personality.get_prompt_context()}")
    print("=====================================\n")
    
    st.markdown("---")
    
    # Profile Summary
    st.markdown("### 📋 Profile Summary")
    st.info(f"""
    **Risk Level:** {risk_tolerance.title()}
    **Timeline:** {time_horizon.replace('_', ' ').title()}
    **Strategy:** {investment_style.title()}
    """)
    
    # Clear chat button with styling
    if st.button("🔄 Reset Conversation", key="clear_chat"):
        print("\n=== DEBUG: Clearing Chat History ===")
        st.session_state.messages = [{"role": "assistant", "content": "👋 Welcome to FinSage AI! How can I help you make data-driven financial decisions today?"}]
        message_history.clear()
        st.rerun()

# Chat input and processing
if prompt := st.chat_input("Ask me about financial analysis, market trends, or investment strategies..."):
    print("\n=== DEBUG: New Chat Input ===")
    print(f"Prompt: {prompt}")
    
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        response_container = st.container()
        callback_handler = CustomStreamlitCallbackHandler(parent_container=response_container)
        
        try:
            print("\n=== DEBUG: Processing User Input ===")
            settings = {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
            }
            
            state = {   
            "current_date": datetime.now(),
            "messages": list(message_history.messages) + [prompt],
            "user_input": prompt,
            "config": settings,
            "callback": callback_handler,
            "personality": st.session_state.personality,
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
            "agent_tools": [],  
            "date_available": "",
            "relevant_tables": {
                "tables": [],
                "explanation": ""
            },
            "response": "",  
            "wrong_generated_queries": [],
            "wrong_formatted_results": []
        }
        }
            
            debug_state(state)
            
            print("\n=== DEBUG: Invoking Flow Graph ===")
            output = FinSage_agent.invoke(
                state,
                {"recursion_limit": 30},
            )
            print("Flow graph execution completed")
            
            print("\n=== DEBUG: Processing Output ===")
            process_agent_output(output, response_container)
            
            final_message = next(
                (msg for msg in output.get("messages", []) 
                 if hasattr(msg, 'name') and msg.name == "FinalSynthesis"),
                None
            )
            if final_message:
                print("Final synthesis found and added to chat history")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_message.content
                })
            
            # Move the evaluation expanders inside the try block where output is defined
            with st.expander("🔍 Tool Usage Evaluation"):
                st.markdown("### News Sentiment Agent Tool Usage:")
                st.dataframe(get_all_tools_called_eval_df(output, "news_sentiment_agent_internal_state")) 
                st.markdown("### Financial Metrics Agent Tool Usage:")
                st.dataframe(get_all_tools_called_eval_df(output, "financial_metrics_agent_internal_state"))
                st.markdown("### Market Intelligence Agent Tool Usage:")
                st.dataframe(get_all_tools_called_eval_df(output, "market_intelligence_agent_internal_state"))

            with st.expander("📝 Topic Adherence Evaluation"):
                st.markdown("### News Sentiment Agent Topic Adherence:")
                st.dataframe(get_topic_adherence_eval_df(output, "news_sentiment_agent_internal_state"))
                st.markdown("### Financial Metrics Agent Topic Adherence:")
                st.dataframe(get_topic_adherence_eval_df(output, "financial_metrics_agent_internal_state"))
                st.markdown("### Market Intelligence Agent Topic Adherence:")
                st.dataframe(get_topic_adherence_eval_df(output, "market_intelligence_agent_internal_state"))
            
            with st.expander(" SQL Agent Evaluation"):
                wrong_generated_queries = output["sql_agent_internal_state"]['wrong_generated_queries']
                wrong_formatted_results = output['sql_agent_internal_state']['wrong_formatted_results']
                data = visualize_sql_agent_performance(
                    query_errors={"data": wrong_generated_queries, "title": "Query Generation Issues"},
                    format_errors={"data": wrong_formatted_results, "title": "Formatting Problems"}
                    )
                st.table(data)

            # with st.expander("💬 Agent Messages"):
            #     for agent_state in ["news_sentiment_agent_internal_state", 
            #                         "market_intelligence_agent_internal_state",
            #                         "sql_agent_internal_state"]:
            #         st.write(f"### {agent_state.replace('_internal_state', '').title()}")
            #         messages = response_container[agent_state].get("messages", [])
            #         for msg in messages:
            #             st.text(f"{msg.get('role', 'unknown')}: {msg.get('content', 'No content')}")

            with st.expander("🛠️ Raw Response Data"):
                st.json(response_container)
            
        except Exception as e:
            print(f"\n=== DEBUG: Error Occurred ===\n{str(e)}")
            st.error("🚨 An error occurred. Please try again or rephrase your question.")
            print(f"Detailed error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"I apologize, but I encountered an error: {str(e)}"
            })