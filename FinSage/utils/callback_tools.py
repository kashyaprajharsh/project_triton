from typing import Dict, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


class CustomConsoleCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        """Initialize the handler"""
        self.current_agent_name = None
        super().__init__()

    def write_agent_name(self, name: str):
        """Display agent name"""
        self.current_agent_name = name
        print(f"\n=== Agent: {name} ===")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Display tool execution start"""
        print(f"\n🔧 Using tool: {serialized['name']}")

    def on_tool_end(self, output: str, **kwargs):
        """Display tool execution result"""
        print("\n📤 Tool output:")
        print("-" * 50)
        print(output)
        print("-" * 50)

    def on_agent_action(self, action: AgentAction, **kwargs):
        """Display agent action"""
        print(f"\n🎯 Action: {action.tool}")
        print("Input:")
        print("-" * 50)
        print(action.tool_input)
        print("-" * 50)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs):
        """Display when LLM starts processing"""
        print("\n🤔 Processing...")

    def on_llm_end(self, response, **kwargs):
        """Display final LLM response"""
        if hasattr(response, 'generations') and response.generations:
            print("\n Final LLM Response:")
            print("-" * 50)
            print(response.generations[0][0].text)
            print("-" * 50)

    def on_tool_error(self, error: str, **kwargs):
        """Display tool errors"""
        print(f"\n❌ Error: {error}")



class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, parent_container):
        """Initialize the handler with a parent container"""
        self._parent_container = parent_container
        self.current_agent_container = None
        self.is_finish_node = False
        super().__init__()

    def write_agent_name(self, name: str):
        """Create a new expander for each agent"""
        self.is_finish_node = (name == "Conversation Handler 💬")
        if not self.is_finish_node:
            self.current_agent_container = self._parent_container.expander(name, expanded=True)
        else:
            self.current_agent_container = self._parent_container

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Display tool execution start"""
        if self.current_agent_container:
            self.current_agent_container.markdown(f"🔧 Using tool: **{serialized['name']}**")

    def on_tool_end(self, output: str, **kwargs):
        """Display tool execution result"""
        if self.current_agent_container:
            if self.is_finish_node:
                # Direct output for finish node
                self.current_agent_container.markdown(output)
            else:
                # Regular tool output handling with formatting
                if isinstance(output, str) and "SQL Query:" in output:
                    # Split SQL results into query and results sections
                    parts = output.split("Results:", 1)
                    if len(parts) == 2:
                        query = parts[0].replace("SQL Query:", "").strip()
                        results = parts[1].strip()
                        
                        self.current_agent_container.markdown("📝 **SQL Query:**")
                        self.current_agent_container.code(query, language="sql")
                        self.current_agent_container.markdown("📊 **Results:**")
                        self.current_agent_container.markdown(results)
                    else:
                        self.current_agent_container.code(output)
                else:
                    self.current_agent_container.markdown("📤 Tool output:")
                    self.current_agent_container.code(output)

    def on_agent_action(self, action: AgentAction, **kwargs):
        """Display agent action"""
        if self.current_agent_container:
            self.current_agent_container.markdown(f"🎯 Action: **{action.tool}**")
            self.current_agent_container.markdown("Input:")
            self.current_agent_container.code(action.tool_input)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs):
        """Display when LLM starts processing"""
        if self.current_agent_container:
            self.current_agent_container.markdown("🤔 Processing...")

    def on_llm_end(self, response, **kwargs):
        """Display final LLM response"""
        if self.current_agent_container and hasattr(response, 'generations') and response.generations:
            self.current_agent_container.markdown(response.generations[0][0].text)

    def on_tool_error(self, error: str, **kwargs):
        """Display tool errors"""
        if self.current_agent_container:
            self.current_agent_container.error(f"Error: {error}")