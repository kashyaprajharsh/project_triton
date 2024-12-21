from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List

# Local imports
from FinSage.config.members import get_team_members_details
from FinSage.prompts.system_prompts import get_supervisor_prompt_template, get_finish_step_prompt
from FinSage.models.schemas import RouteSchema


def get_supervisor_chain(llm: BaseChatModel, current_date=None):
    """
    Returns a supervisor chain that manages a conversation between workers.

    The supervisor chain is responsible for managing a conversation between a group
    of workers. It prompts the supervisor to select the next worker to act, and
    each worker performs a task and responds with their results and status. The
    conversation continues until the supervisor decides to finish.

    Returns:
        supervisor_chain: A chain of prompts and functions that handle the conversation
                          between the supervisor and workers.
    """

    team_members = get_team_members_details()
    
    date_context = f"\nCurrent Analysis Date: {current_date.strftime('%Y-%m-%d %H:%M:%S UTC') if current_date else 'Not specified'}"
    
    formatted_string = ""
    for i, member in enumerate(team_members):
        formatted_string += (
            f"**{i+1} {member['name']}**\nRole: {member['description']}\n\n"
        )

    # Remove the trailing new line
    formatted_members_string = formatted_string.strip()

    # # Debug prints to verify variables
    # print("\n=== Supervisor Chain Variables ===")
    # print(f"Date Context: {date_context}")
    # print(f"\nTeam Members:\n{formatted_members_string}")
    
    options = [member["name"] for member in team_members]
    # print(f"\nAvailable Options: {options}")

    system_prompt = get_supervisor_prompt_template()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                """Based on the conversation history and investment profile, determine the next step.
                
                For simple queries like "fundamentals", select a single primary agent.
                For complex queries, coordinate multiple agents as needed.
            
            TASK ASSIGNMENT REQUIRED:
            1. Select the next agent or action
            2. Provide detailed task description
            3. Define expected outputs
            4. List validation criteria
            5. Specify query_type as one of: 'financial_analysis', 'non_financial_analysis'
            Available options: {options}
            IMPORTANT ROUTING RULES:
                Max_attempts = 1 for each agent
                1. Do not route to an agent again and again that has already succeeded and all tools are called and output is generated(marked as completed)
            
            Respond with your routing decision."""
            )
        ]
    ).partial(
        options=str(options), 
        members=formatted_members_string, 
        date_context=date_context,
        personality="{personality}"
    )

    # Debug the final formatted prompt template
    # print("\n=== Final Prompt Template ===")
    # print("Variables being passed:")
    # print(f"- options: {str(options)}")
    # print(f"- members: [length: {len(formatted_members_string)} chars]")
    # print(f"- date_context: {date_context}")

    supervisor_chain = prompt | llm.with_structured_output(RouteSchema)

    return supervisor_chain



def get_finish_chain(llm: BaseChatModel):
    """
    If the supervisor decides to finish the conversation, this chain is executed.
    """
    system_prompt = get_finish_step_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("system", system_prompt),
            ("human", "Please provide an appropriate response to the conversation.")
        ]
    )
    finish_chain = prompt | llm
    return finish_chain