import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
from langchain_core.messages import HumanMessage, AIMessage

from config import setup_environment

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

# Chat input and processing
if prompt := st.chat_input("What would you like to analyze?"):
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
            
            # Process through agent graph
            output = flow_graph.invoke(
                {
                    "messages": list(message_history.messages) + [prompt],
                    "user_input": prompt,
                    "config": settings,
                    "callback": callback_handler,
                },
                {"recursion_limit": 30},
            )
            
            # Process and display the output
            process_agent_output(output, response_container)
            
            # Add final synthesis to chat history
            final_message = next(
                (msg for msg in output.get("messages", []) 
                 if hasattr(msg, 'name') and msg.name == "FinalSynthesis"),
                None
            )
            if final_message:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_message.content
                })
            
        except Exception as e:
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