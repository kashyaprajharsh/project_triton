import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from config import setup_environment
from personality import AgentPersonality, RiskTolerance, TimeHorizon, InvestmentStyle

setup_environment()

# Page configuration        
st.set_page_config(layout="wide")
st.title("Financial Analysis Assistant 📊")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your financial analysis assistant. How can I help you analyze financial data today?"}
    ]

# Create the agent flow
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def process_agent_output(output, response_container):
    """Process and display the agent output in a structured way"""
    try:
        messages = output.get("messages", [])
        final_synthesis = None
        
        # Create an expander for the final synthesis
        for message in messages:
            if hasattr(message, 'name') and message.name == "FinalSynthesis":
                final_synthesis = message.content
                break
        
        if final_synthesis:
            with response_container.expander("🎯 Final Recommendation", expanded=True):
                response_container.markdown(final_synthesis)
                
    except Exception as e:
        st.error(f"Error processing output: {str(e)}")

with st.sidebar:
    st.header("Investment Profile")
    
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        options=[rt.value for rt in RiskTolerance],
        index=[rt.value for rt in RiskTolerance].index(RiskTolerance.MODERATE.value),
        help="Choose your risk tolerance level"
    )
    
    time_horizon = st.selectbox(
        "Time Horizon",
        options=[th.value for th in TimeHorizon],
        index=[th.value for th in TimeHorizon].index(TimeHorizon.MEDIUM_TERM.value),
        help="Select your investment time horizon"
    )
    
    investment_style = st.selectbox(
        "Investment Style",
        options=[style.value for style in InvestmentStyle],
        index=[style.value for style in InvestmentStyle].index(InvestmentStyle.BLEND.value),
        help="Choose your preferred investment style"
    )
    
    # Initialize or update personality
    personality = AgentPersonality(
        risk_tolerance=RiskTolerance(risk_tolerance),
        time_horizon=TimeHorizon(time_horizon),
        investment_style=InvestmentStyle(investment_style)
    )
    
    # Always update the personality in session state
    st.session_state.personality = personality
    
    # Debug print
    print("\n=== DEBUG: Current Investment Profile ===")
    print(f"Risk Tolerance: {risk_tolerance}")
    print(f"Time Horizon: {time_horizon}")
    print(f"Investment Style: {investment_style}")
    print(f"Personality Context: {st.session_state.personality.get_prompt_context()}")
    print("=====================================\n")

# Add he top after imports
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

# Chat input and processing
if prompt := st.chat_input("What would you like to analyze?"):
    print("\n=== DEBUG: New Chat Input ===")
    print(f"Prompt: {prompt}")
    
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_container = st.container()
        callback_handler = CustomStreamlitCallbackHandler(parent_container=response_container)
        
        try:
            # Default settings for the model
            settings = {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
            }
            
            # Create state and debug it
            state = {   
                "current_date": datetime.now(),
                "messages": list(message_history.messages) + [prompt],
                "user_input": prompt,
                "config": settings,
                "callback": callback_handler,
                "personality": st.session_state.personality
            }
            debug_state(state)
            
            # Process through agent graph
            print("\n=== DEBUG: Invoking Flow Graph ===")
            output = flow_graph.invoke(
                state,
                {"recursion_limit": 30},
            )
            print("Flow graph execution completed")
            
            # Process and display the output
            print("\n=== DEBUG: Processing Output ===")
            process_agent_output(output, response_container)
            
            # Add final synthesis to chat history
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
            
        except Exception as e:
            print(f"\n=== DEBUG: Error Occurred ===\n{str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I apologize, but I encountered an error. Please try again."
            })

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your financial analysis assistant. How can I help you analyze financial data today?"}]
    message_history.clear()
    st.rerun()