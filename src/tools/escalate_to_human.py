# Import Third-party libraries
from langchain_core.tools import tool

ESCALATION_MESSAGE = (
    "I'm not permitted to fulfil this request myself. I'm escalating it to our "
    "specialist team, and a representative will be in touch with you shortly."
)


@tool("escalate_to_human")
def escalate_to_human() -> str:
    """
    Hand the request to the bank's specialist team.

    Use this when the request is transactional or when they are upset, disputing
    something, or the situation is otherwise sensitive.

    Returns the escalation message.
    """
    return ESCALATION_MESSAGE
