from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List

from members import get_team_members_details
from prompts import get_supervisor_prompt_template, get_finish_step_prompt
from schemas import RouteSchema


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
    formatted_members_string = "".join(
        f"**{i+1} {member['name']}**\nRole: {member['description']}\n\n"
        for i, member in enumerate(team_members)
    ).strip()

    system_prompt = get_supervisor_prompt_template(current_date)
    options = [member["name"] for member in team_members] + ["Synthesizer", "FINISH"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", """
            Given the conversation above and investment profile, determine the next step:
            1. If reflection analysis was just received:
               - Review reflection recommendations
               - Route to specific agent if gaps identified
               - Route to Synthesizer if analysis is complete
            2. If synthesis is complete, select FINISH
            3. Otherwise, select appropriate next agent 
            
            Select one of: {options}
            """),
    ]).partial(
        options=str(options),
        members=formatted_members_string
    )
    
    return prompt | llm.with_structured_output(RouteSchema)


def get_finish_chain(llm: BaseChatModel):
    """
    If the supervisor decides to finish the conversation, this chain is executed.
    """
    system_prompt = get_finish_step_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("system", system_prompt),
        ]
    )
    finish_chain = prompt | llm
    return finish_chain


