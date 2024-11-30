from typing import Any, Dict
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, parent_container):
        """Initialize the handler with a parent container"""
        self._parent_container = parent_container
        self.current_agent_container = None
        super().__init__()

    def write_agent_name(self, name: str):
        """Create a new expander for each agent"""
        self.current_agent_container = self._parent_container.expander(name, expanded=True)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Display tool execution start"""
        if self.current_agent_container:
            self.current_agent_container.markdown(f"🔧 Using tool: **{serialized['name']}**")

    def on_tool_end(self, output: str, **kwargs):
        """Display tool execution result"""
        if self.current_agent_container:
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